"""Numerical kernels for stationary and decaying turbulence models."""

from __future__ import annotations

import time
import warnings
from functools import lru_cache
from multiprocessing import Pool

import mpmath as mp
import numpy as np
from scipy import integrate, special

from .mpi import gather_grid, get_mpi_context, split_row_indices


def _emit_status(status, message: str, *, force: bool = False) -> None:
    if status is None:
        return
    if hasattr(status, "__call__"):
        try:
            status(message, force=force)
        except TypeError:
            status(message)


class LiveStatusLogger:
    """Notebook-friendly throttled logger for long-running integrations."""

    def __init__(
        self,
        prefix: str = "decaying",
        every_seconds: float = 1.0,
        rank: int = 0,
        root_only: bool = True,
    ):
        self.prefix = prefix
        self.every_seconds = every_seconds
        self.rank = rank
        self.root_only = root_only
        self._last_emit = 0.0

    def __call__(self, message: str, *, force: bool = False) -> None:
        if self.root_only and self.rank != 0:
            return
        now = time.time()
        if force or (now - self._last_emit) >= self.every_seconds:
            label = self.prefix if self.root_only else f"{self.prefix}:rank{self.rank}"
            print(f"[{label}] {message}", flush=True)
            self._last_emit = now


def kernel_bracket(p: float, x: float, y: float) -> float:
    xp32 = x**1.5
    yp32 = y**1.5
    # Match Gogoberidze et al. (2007) Appendix A normalization:
    # Eq. (A4) uses the kernel with an overall factor 2 relative to the
    # reduced form that would start with 27.
    return (
        54.0
        - 2.0 * p**2 * xp32
        - 2.0 * p**2 * yp32
        + p**4 * xp32 * yp32
        + x**(-1.5) * yp32
        + y**(-1.5) * xp32
    )


def integrand_y(y: float, x: float, p: float, q: float, M: float) -> float:
    s = x + y
    if s <= 0:
        return 0.0
    pref = y**0.75 * s**(-0.5) * x**0.75
    bracket = kernel_bracket(p, x, y)
    expo = np.exp(-2.0 * x * y / s * q**2 / M**2)
    erfc_factor = special.erfc(-np.sqrt(2.0) * q / (M * np.sqrt(s)))
    return pref * bracket * expo * erfc_factor


def g_decaying(z):
    """Dimensionless temporal kernel for the decaying-spectrum model."""
    if np.isscalar(z):
        return _g_decaying_scalar(complex(z))
    array = np.asarray(z)
    values = np.empty(array.shape, dtype=complex)
    for index, value in np.ndenumerate(array):
        values[index] = _g_decaying_scalar(complex(value))
    return values


@lru_cache(maxsize=1024)
def _g_decaying_scalar(z: complex) -> complex:
    # For E(t) ∝ (1 + t/tau_1)^(-2/3) with t >= 0 and Fourier convention
    # exp(-i omega t), derivation.tex gives
    # g(q) = exp(i q) * (-i q)^(-5/3) * Gamma(1/3, -i q),
    # where Gamma(1/3, -i q) is the lower incomplete gamma.
    arg = -1j * z
    gamma_lower = mp.gammainc(1.0 / 3.0, arg)
    return complex(mp.e ** (1j * z) * (arg ** (-5.0 / 3.0)) * gamma_lower)


def _cosine_grid(lower: float, upper: float, count: int) -> np.ndarray:
    """Cosine-spaced sample grid on [lower, upper]."""
    u = np.linspace(0.0, 1.0, count)
    return lower + 0.5 * (upper - lower) * (1.0 - np.cos(np.pi * u))


def _conv_intervals(
    q: float, q_bound: float, split_width: float
) -> list[tuple[float, float]]:
    """Sub-intervals for the truncated convolution on [-q_bound, q_bound]."""
    lower = -q_bound
    upper = q_bound
    singular_windows = []
    for point in sorted((0.0, q)):
        window_lo = max(lower, point - split_width)
        window_hi = min(upper, point + split_width)
        if window_lo < window_hi:
            singular_windows.append((window_lo, window_hi))

    pieces = []
    cursor = lower
    for window_lo, window_hi in singular_windows:
        if window_lo > cursor:
            pieces.append((cursor, window_lo))
        cursor = max(cursor, window_hi)
    if cursor < upper:
        pieces.append((cursor, upper))

    deduped = []
    for left, right in pieces:
        if right - left > 0:
            deduped.append((left, right))
    return deduped


def integrand_y_decaying(
    y: float,
    x: float,
    p: float,
    q: float,
    M: float,
    epsabs: float = 2e-3,
    epsrel: float = 1e-2,
    convolution_method: str = "trapz",
    convolution_points: int = 160,
) -> float:
    s = x + y
    if s <= 0:
        return 0.0
    pref = y**0.75 * s**(-0.5) * x**0.75
    bracket = kernel_bracket(p, x, y)
    if abs(bracket) < 1e-12 or abs(pref) < 1e-14:
        return 0.0

    scale = np.sqrt(x * y) / M if M > 0 else 1.0
    q_bound = max(abs(q) + 10.0 * scale, 20.0)
    split_width = max(1e-8, 1e-6 * max(1.0, abs(q)))

    def conv_integrand(q1: float) -> float:
        z1 = q1 * np.sqrt(x) / M
        z2 = (q - q1) * np.sqrt(y) / M
        # The kernel is integrably singular at q1=0 and q1=q. Avoid evaluating
        # exactly at those points while still resolving the nearby behavior.
        if z1 == 0:
            z1 = (split_width * np.sqrt(x) / M) * 1j
        if z2 == 0:
            z2 = (split_width * np.sqrt(y) / M) * 1j
        return (g_decaying(z1) * g_decaying(z2)).real

    try:
        conv_val = 0.0
        if convolution_method == "trapz":
            for lower, upper in _conv_intervals(q, q_bound, split_width):
                if lower >= upper:
                    continue
                q1_values = _cosine_grid(lower, upper, max(convolution_points, 32))
                z1 = q1_values * np.sqrt(x) / M
                z2 = (q - q1_values) * np.sqrt(y) / M
                values = (g_decaying(z1) * g_decaying(z2)).real
                conv_val += np.trapz(values, q1_values)
        elif convolution_method == "quad":
            for lower, upper in _conv_intervals(q, q_bound, split_width):
                if lower >= upper:
                    continue
                part, _ = integrate.quad(
                    conv_integrand,
                    lower,
                    upper,
                    epsabs=epsabs,
                    epsrel=epsrel,
                    limit=100,
                )
                conv_val += part
        else:
            raise ValueError(f"Unknown convolution_method: {convolution_method}")
    except ValueError:
        raise
    except Exception as exc:
        warnings.warn(
            f"Convolution integration failed at x={x:.3e}, y={y:.3e}, q={q:.3e}: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )
        conv_val = 0.0

    return pref * bracket * conv_val


def _integration_bounds(x: float, p: float, R: float) -> tuple[float, float] | None:
    tilde_k1 = x**(-0.75)
    u_min = max(abs(tilde_k1 - p), 1.0)
    u_max = min(tilde_k1 + p, R**0.75)
    if not (u_min < u_max):
        return None
    y_min = u_max ** (-4.0 / 3.0)
    y_max = u_min ** (-4.0 / 3.0)
    if not (y_min < y_max):
        return None
    return y_min, y_max


def inner_integral(
    x: float,
    p: float,
    q: float,
    M: float,
    R: float,
    epsabs: float,
    epsrel: float,
) -> float:
    bounds = _integration_bounds(x, p, R)
    if bounds is None:
        return 0.0
    y_min, y_max = bounds
    value, _ = integrate.quad(
        integrand_y,
        y_min,
        y_max,
        args=(x, p, q, M),
        epsabs=epsabs,
        epsrel=epsrel,
        limit=200,
    )
    return value


def inner_integral_decaying(
    x: float,
    p: float,
    q: float,
    M: float,
    R: float,
    epsabs: float,
    epsrel: float,
    convolution_method: str,
    convolution_points: int,
    integration_method: str,
    y_points: int,
) -> float:
    bounds = _integration_bounds(x, p, R)
    if bounds is None:
        return 0.0
    y_min, y_max = bounds
    if integration_method == "sampled":
        y_values = np.geomspace(y_min, y_max, max(y_points, 16))
        values = np.array(
            [
                integrand_y_decaying(
                    yy,
                    x,
                    p,
                    q,
                    M,
                    epsabs,
                    epsrel,
                    convolution_method,
                    convolution_points,
                )
                for yy in y_values
            ]
        )
        return float(np.trapz(values, y_values))
    value, _ = integrate.quad(
        integrand_y_decaying,
        y_min,
        y_max,
        args=(x, p, q, M, epsabs, epsrel, convolution_method, convolution_points),
        epsabs=epsabs,
        epsrel=epsrel,
        limit=100,
    )
    return value


def _h_prefactor(p: float, M: float, k0: float) -> float:
    p_floor = max(p, 1e-10)
    return 3.0 * M**3 * k0**(-4) / (256.0 * (2.0 * np.pi) ** 1.5 * p_floor)


def H_pq(
    p: float,
    q: float,
    M: float = 1.0,
    R: float = 1e6,
    k0: float = 1.0,
    epsabs: float = 1e-8,
    epsrel: float = 1e-6,
) -> float:
    p_floor = max(p, 1e-10)
    x_lo = R**(-1)
    x_hi = 1.0

    def outer_x(xx: float) -> float:
        return inner_integral(xx, p_floor, q, M, R, epsabs, epsrel)

    value, _ = integrate.quad(outer_x, x_lo, x_hi, epsabs=epsabs, epsrel=epsrel, limit=200)
    return _h_prefactor(p_floor, M, k0) * value


def H_pq_decaying(
    p: float,
    q: float,
    M: float = 1.0,
    R: float = 1e6,
    k0: float = 1.0,
    epsabs: float = 2e-3,
    epsrel: float = 1e-2,
    convolution_method: str = "trapz",
    convolution_points: int = 160,
    integration_method: str = "sampled",
    x_points: int = 24,
    y_points: int = 24,
    status=None,
) -> float:
    p_floor = max(p, 1e-10)
    x_lo = R**(-1)
    x_hi = 1.0

    _emit_status(
        status,
        (
            f"start H_pq_decaying p={p:.3e} q={q:.3e} M={M:.3e} "
            f"conv={convolution_method} integ={integration_method}"
        ),
        force=True,
    )

    def outer_x(xx: float) -> float:
        return inner_integral_decaying(
            xx,
            p_floor,
            q,
            M,
            R,
            epsabs,
            epsrel,
            convolution_method,
            convolution_points,
            integration_method,
            y_points,
        )

    if integration_method == "sampled":
        x_values = np.geomspace(x_lo, x_hi, max(x_points, 16))
        x_integrand = np.zeros_like(x_values)
        for index, xx in enumerate(x_values):
            x_integrand[index] = outer_x(xx)
            _emit_status(
                status,
                f"  x-step {index + 1}/{len(x_values)} for p={p:.3e}, q={q:.3e}",
            )
        value = float(np.trapz(x_integrand, x_values))
    else:
        value, _ = integrate.quad(outer_x, x_lo, x_hi, epsabs=epsabs, epsrel=epsrel, limit=80)
    result = _h_prefactor(p_floor, M, k0) * value
    _emit_status(
        status,
        f"done H_pq_decaying p={p:.3e} q={q:.3e} -> {result:.3e}",
        force=True,
    )
    return result


def _compute_decaying_row(task):
    row_index, q, ps, kwargs = task
    row_values = np.zeros(len(ps))
    for j, p in enumerate(ps):
        try:
            row_values[j] = H_pq_decaying(p, q, **kwargs)
        except Exception as exc:
            warnings.warn(
                f"H_pq_decaying failed at p={p:.3e}, q={q:.3e}: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            row_values[j] = np.nan
    return row_index, row_values


def H_pq_decaying_grid(
    ps,
    qs,
    M: float = 1.0,
    R: float = 1e4,
    k0: float = 1.0,
    epsabs: float = 2e-3,
    epsrel: float = 1e-2,
    verbose: bool = False,
    convolution_method: str = "trapz",
    convolution_points: int = 160,
    integration_method: str = "sampled",
    x_points: int = 24,
    y_points: int = 24,
    status=None,
    log_points: bool = True,
    use_mpi: bool = False,
    processes: int = 1,
) -> np.ndarray:
    ps = np.atleast_1d(ps)
    qs = np.atleast_1d(qs)
    context = get_mpi_context(use_mpi)
    if context is None:
        row_indices = np.arange(len(qs))
        rank = 0
        size = 1
    else:
        row_indices = split_row_indices(len(qs), context.rank, context.size)
        rank = context.rank
        size = context.size
    local_rows: dict[int, np.ndarray] = {}
    if status is not None:
        reporter = status
    elif verbose:
        reporter = LiveStatusLogger(rank=rank, root_only=True)
    else:
        reporter = None

    _emit_status(
        reporter,
        (
            f"grid start rows={len(qs)} cols={len(ps)} M={M:.3e} method={convolution_method} "
            f"points={convolution_points} rank={rank + 1}/{size}"
        ),
        force=True,
    )

    decaying_kwargs = {
        "M": M,
        "R": R,
        "k0": k0,
        "epsabs": epsabs,
        "epsrel": epsrel,
        "convolution_method": convolution_method,
        "convolution_points": convolution_points,
        "integration_method": integration_method,
        "x_points": x_points,
        "y_points": y_points,
        "status": None,
    }

    if context is None and processes > 1:
        _emit_status(reporter, f"using multiprocessing with {processes} workers", force=True)
        tasks = [(i, qs[i], ps, decaying_kwargs) for i in row_indices]
        with Pool(processes=min(processes, len(tasks))) as pool:
            for row_index, row_values in pool.imap_unordered(_compute_decaying_row, tasks):
                local_rows[row_index] = row_values
                _emit_status(reporter, f"row {row_index + 1}/{len(qs)} complete", force=True)
    else:
        for i in row_indices:
            q = qs[i]
            row_start = time.time()
            _emit_status(reporter, f"row {i + 1}/{len(qs)} q={q:.4e}", force=True)
            row_values = np.zeros(len(ps))
            for j, p in enumerate(ps):
                point_status = reporter if log_points else None
                _emit_status(
                    reporter,
                    f"point {j + 1}/{len(ps)} row={i + 1} p={p:.4e} q={q:.4e}",
                    force=True,
                )
                try:
                    row_values[j] = H_pq_decaying(
                        p,
                        q,
                        M=M,
                        R=R,
                        k0=k0,
                        epsabs=epsabs,
                        epsrel=epsrel,
                        convolution_method=convolution_method,
                        convolution_points=convolution_points,
                        integration_method=integration_method,
                        x_points=x_points,
                        y_points=y_points,
                        status=point_status,
                    )
                except Exception as exc:
                    _emit_status(
                        reporter,
                        f"error at row={i + 1} col={j + 1} p={p:.4e} q={q:.4e}: {exc}",
                        force=True,
                    )
                    row_values[j] = np.nan
            local_rows[i] = row_values
            _emit_status(
                reporter,
                f"row {i + 1}/{len(qs)} complete in {time.time() - row_start:.1f}s",
                force=True,
            )

    if context is None:
        grid = np.zeros((len(qs), len(ps)))
        for row_index, row_values in local_rows.items():
            grid[row_index] = row_values
        _emit_status(reporter, "grid complete", force=True)
        return grid

    grid = gather_grid(local_rows, (len(qs), len(ps)), context)
    context.comm.Barrier()
    if context.rank == 0:
        _emit_status(reporter, "grid complete", force=True)
    return grid


def H_k0_analytic(q, M: float = 1.0, k0: float = 1.0, R: float = 1e4):
    q_array = np.atleast_1d(q)
    output = np.zeros_like(q_array, dtype=float)

    x_min = R**(-1)
    x_max = 1.0
    prefactor = 7.0 * M**3 * k0**(-4) / (16.0 * np.pi**1.5)

    for index, q_value in enumerate(q_array):
        if q_value <= 0:
            output[index] = np.nan
            continue

        q_bar = q_value / M

        def integrand_x(x: float) -> float:
            return x ** (11.0 / 4.0) * np.exp(-q_bar**2 * x) * special.erfc(-q_bar * np.sqrt(x))

        try:
            integral, _ = integrate.quad(
                integrand_x,
                x_min,
                x_max,
                epsabs=1e-10,
                epsrel=1e-8,
                limit=300,
            )
        except ValueError:
            raise
        except Exception as exc:
            warnings.warn(
                f"H_k0_analytic integration failed at q={q_value:.3e}: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            integral = 0.0
        output[index] = prefactor * integral

    if np.isscalar(q):
        return output[0]
    return output


# ── Delta-k (monochromatic E = E0 * delta(k - k0)) models ──────────────────


def K0_p(p: float) -> float:
    """Geometric kernel evaluated at k1 = u = k0 for the monochromatic spectrum.

    K0(p) = 14/3 - p^2/3 + p^4/12,  p = k/k0.

    Derived by setting k1 = u = k0 in the full kernel K(k, k1, u).  The
    triangle inequality forces p <= 2 for the result to be nonzero.
    """
    return 14.0 / 3.0 - p**2 / 3.0 + p**4 / 12.0


def H_delta_k_kraichnan(p, Omega):
    """Dimensionless GW kernel for E = E0*delta(k-k0) with Kraichnan decorrelation.

    Fully closed-form (no numerical integration required):

        H(p, Omega) = K0(p)/p * Theta(2-p) * exp(-Omega^2 / (2*pi))

    where p = k/k0 and Omega = omega/eta0.  The Heaviside Theta(2-p) reflects
    the triangle-inequality constraint k <= 2*k0.

    Parameters
    ----------
    p, Omega : scalar or array-like
    """
    p = np.asarray(p, dtype=float)
    Omega = np.asarray(Omega, dtype=float)
    result = np.where(
        (p > 0) & (p <= 2.0),
        K0_p(p) / np.maximum(p, 1e-30) * np.exp(-Omega**2 / (2.0 * np.pi)),
        0.0,
    )
    return float(result) if result.ndim == 0 else result


def _temporal_conv_decay(
    q: float,
    q_bound: float | None = None,
    split_width: float | None = None,
    n_points: int = 200,
) -> float:
    """1-D convolution  int dq1 g(q1) * g(q - q1)  for the decay temporal model.

    g is the dimensionless decay kernel from g_decaying.  This is the pure
    temporal factor shared by H_delta_k_decay and H_white_decay.
    """
    if q_bound is None:
        q_bound = max(abs(q) + 20.0, 30.0)
    if split_width is None:
        split_width = max(1e-8, 1e-6 * max(1.0, abs(q)))
    conv_val = 0.0
    for lower, upper in _conv_intervals(q, q_bound, split_width):
        if lower >= upper:
            continue
        q1_vals = _cosine_grid(lower, upper, max(n_points, 32))
        vals = (g_decaying(q1_vals) * g_decaying(q - q1_vals)).real
        conv_val += np.trapz(vals, q1_vals)
    return conv_val


def H_delta_k_decay(
    p: float,
    q: float,
    n_points: int = 200,
) -> float:
    """Dimensionless GW kernel for E = E0*delta(k-k0) with decay temporal model.

        H(p, q) = K0(p)/p * Theta(2-p) * int dq1 g(q1)*g(q-q1)

    where q = omega*tau1 and g is the decay kernel (g_decaying).

    The spatial delta functions collapse k1 and u to k0, leaving only the
    1-D temporal convolution as a numerical task.
    """
    if p <= 0 or p > 2.0:
        return 0.0
    return K0_p(p) / p * _temporal_conv_decay(q, n_points=n_points)


# ── White-noise (R_ij ∝ delta^3(r)) models ─────────────────────────────────


def kernel_bracket_zy(p: float, z: float, y: float) -> float:
    """Geometric kernel in (z = k1/k0, y = u/k0, p = k/k0) variables.

    Identical in value to kernel_bracket but using direct (z,y) variables
    instead of the Gogoberidze (x,y) substitution.  Returns 12*K(k,k1,u).
    """
    return (
        54.0
        - 2.0 * p**2 / z**2
        - 2.0 * p**2 / y**2
        + p**4 / (z**2 * y**2)
        + z**2 / y**2
        + y**2 / z**2
    )


def _white_spatial_integral(
    p: float,
    R: float,
    epsabs: float = 1e-4,
    epsrel: float = 1e-3,
) -> float:
    """2-D wavevector integral for the white-noise spatial model.

    S(p, R) = int_1^{R^{3/4}} (z/p) dz  int_{max(|p-z|,1)}^{min(p+z,R^{3/4})} y dy  K_zy(p,z,y)

    This spatial factor is independent of frequency and is shared by both
    H_white_kraichnan (multiplied by a Gaussian) and H_white_decay (multiplied
    by the temporal convolution).
    """
    Rd34 = R**0.75
    p_floor = max(p, 1e-10)

    def inner_y(y: float, z: float) -> float:
        if y <= 0 or z <= 0:
            return 0.0
        return y * kernel_bracket_zy(p_floor, z, y)

    def outer_z(z: float) -> float:
        y_lo = max(abs(p_floor - z), 1.0)
        y_hi = min(p_floor + z, Rd34)
        if y_lo >= y_hi:
            return 0.0
        val, _ = integrate.quad(
            inner_y, y_lo, y_hi, args=(z,), epsabs=epsabs, epsrel=epsrel, limit=200
        )
        return z / p_floor * val

    result, _ = integrate.quad(outer_z, 1.0, Rd34, epsabs=epsabs, epsrel=epsrel, limit=200)
    return result


def H_white_kraichnan(
    p: float,
    Omega: float,
    R: float = 1e6,
    epsabs: float = 1e-4,
    epsrel: float = 1e-3,
) -> float:
    """Dimensionless GW kernel for delta^3(r) spatial correlations + Kraichnan decorrelation.

        H(p, Omega) = exp(-Omega^2/(2*pi)) * S(p, R)

    where S(p,R) is the 2-D spatial integral (independent of Omega) and
    Omega = omega/eta0.  The temporal factor is Gaussian and fully analytic.
    """
    spatial = _white_spatial_integral(p, R, epsabs=epsabs, epsrel=epsrel)
    return float(np.exp(-Omega**2 / (2.0 * np.pi))) * spatial


def H_white_decay(
    p: float,
    q: float,
    R: float = 1e6,
    epsabs: float = 1e-4,
    epsrel: float = 1e-3,
    n_points: int = 200,
) -> float:
    """Dimensionless GW kernel for delta^3(r) spatial correlations + decay temporal model.

        H(p, q) = S(p, R) * int dq1 g(q1)*g(q-q1)

    The two factors are independent and computed separately, so either can be
    cached when sweeping the other variable.
    """
    spatial = _white_spatial_integral(p, R, epsabs=epsabs, epsrel=epsrel)
    temporal = _temporal_conv_decay(q, n_points=n_points)
    return spatial * temporal


# ── Grid helpers for new models ─────────────────────────────────────────────


def H_delta_k_kraichnan_grid(ps, Omegas) -> np.ndarray:
    """2-D grid of H_delta_k_kraichnan(p, Omega).

    Fully vectorised (no loops): returns array of shape (len(Omegas), len(ps)).
    """
    ps = np.atleast_1d(np.asarray(ps, dtype=float))
    Omegas = np.atleast_1d(np.asarray(Omegas, dtype=float))
    PP, OO = np.meshgrid(ps, Omegas)
    return H_delta_k_kraichnan(PP, OO)


def H_delta_k_decay_grid(
    ps,
    qs,
    n_points: int = 200,
    status=None,
) -> np.ndarray:
    """2-D grid of H_delta_k_decay(p, q).

    Computes the temporal convolution once per q value, then broadcasts over p.
    Returns array of shape (len(qs), len(ps)).
    """
    ps = np.atleast_1d(np.asarray(ps, dtype=float))
    qs = np.atleast_1d(np.asarray(qs, dtype=float))
    grid = np.zeros((len(qs), len(ps)))
    for i, q in enumerate(qs):
        _emit_status(status, f"H_delta_k_decay_grid q={q:.4e} ({i+1}/{len(qs)})", force=True)
        conv = _temporal_conv_decay(q, n_points=n_points)
        for j, p in enumerate(ps):
            grid[i, j] = 0.0 if (p <= 0 or p > 2.0) else K0_p(p) / p * conv
    return grid


def H_white_kraichnan_grid(
    ps,
    Omegas,
    R: float = 1e6,
    epsabs: float = 1e-4,
    epsrel: float = 1e-3,
    status=None,
) -> np.ndarray:
    """2-D grid of H_white_kraichnan(p, Omega).

    Precomputes S(p) for each p once, then multiplies by the Gaussian factor.
    Returns array of shape (len(Omegas), len(ps)).
    """
    ps = np.atleast_1d(np.asarray(ps, dtype=float))
    Omegas = np.atleast_1d(np.asarray(Omegas, dtype=float))
    spatial = np.zeros(len(ps))
    for j, p in enumerate(ps):
        _emit_status(status, f"H_white_kraichnan_grid p={p:.4e} ({j+1}/{len(ps)})", force=True)
        spatial[j] = _white_spatial_integral(p, R, epsabs=epsabs, epsrel=epsrel)
    OO = Omegas[:, np.newaxis]
    return np.exp(-OO**2 / (2.0 * np.pi)) * spatial[np.newaxis, :]


def H_white_decay_grid(
    ps,
    qs,
    R: float = 1e6,
    epsabs: float = 1e-4,
    epsrel: float = 1e-3,
    n_points: int = 200,
    status=None,
) -> np.ndarray:
    """2-D grid of H_white_decay(p, q).

    Precomputes S(p) and the temporal convolution conv(q) independently, then
    forms the outer product.  Returns array of shape (len(qs), len(ps)).
    """
    ps = np.atleast_1d(np.asarray(ps, dtype=float))
    qs = np.atleast_1d(np.asarray(qs, dtype=float))
    spatial = np.zeros(len(ps))
    for j, p in enumerate(ps):
        _emit_status(status, f"H_white_decay_grid spatial p={p:.4e} ({j+1}/{len(ps)})", force=True)
        spatial[j] = _white_spatial_integral(p, R, epsabs=epsabs, epsrel=epsrel)
    temporal = np.zeros(len(qs))
    for i, q in enumerate(qs):
        _emit_status(
            status, f"H_white_decay_grid temporal q={q:.4e} ({i+1}/{len(qs)})", force=True
        )
        temporal[i] = _temporal_conv_decay(q, n_points=n_points)
    return temporal[:, np.newaxis] * spatial[np.newaxis, :]
