"""Reusable numerical tools for gravitational-wave turbulence studies."""
from .core import (
    # Kolmogorov stationary (Kraichnan) model
    H_k0_analytic,
    H_pq,
    # Kolmogorov decaying model
    H_pq_decaying,
    H_pq_decaying_grid,
    # Monochromatic spectrum E = E0*delta(k-k0)
    K0_p,
    H_delta_k_kraichnan,
    H_delta_k_kraichnan_grid,
    H_delta_k_decay,
    H_delta_k_decay_grid,
    # White-noise (delta^3(r)) spatial correlations
    kernel_bracket_zy,
    H_white_kraichnan,
    H_white_kraichnan_grid,
    H_white_decay,
    H_white_decay_grid,
    # Shared utilities
    LiveStatusLogger,
    g_decaying,
    kernel_bracket,
)
from .mpi import get_mpi_context, mpi_is_active
from .plotting import (
    example_scan_and_plot,
    plot_gogoberidze_2007_figure1,
    plot_Hqq_decaying,
    plot_p0_spectra_params,
    plot_scans_for_M_list,
    plot_spectra_M,
    plot_spectra_M_analytic,
    scan_and_plot_grid,
)

__all__ = [
    # Kolmogorov stationary (Kraichnan) model
    "H_k0_analytic",
    "H_pq",
    # Kolmogorov decaying model
    "H_pq_decaying",
    "H_pq_decaying_grid",
    # Monochromatic spectrum E = E0*delta(k-k0)
    "K0_p",
    "H_delta_k_kraichnan",
    "H_delta_k_kraichnan_grid",
    "H_delta_k_decay",
    "H_delta_k_decay_grid",
    # White-noise (delta^3(r)) spatial correlations
    "kernel_bracket_zy",
    "H_white_kraichnan",
    "H_white_kraichnan_grid",
    "H_white_decay",
    "H_white_decay_grid",
    # Shared utilities
    "LiveStatusLogger",
    "example_scan_and_plot",
    "g_decaying",
    "get_mpi_context",
    "kernel_bracket",
    "mpi_is_active",
    "plot_gogoberidze_2007_figure1",
    "plot_Hqq_decaying",
    "plot_p0_spectra_params",
    "plot_scans_for_M_list",
    "plot_spectra_M",
    "plot_spectra_M_analytic",
    "scan_and_plot_grid",
]
