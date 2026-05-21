"""Project-wide matplotlib style for Polarization-of-Gravitational-Waves.

Usage
-----
    from gw_turbulence.plot_style import (
        apply_paper_style, FIGSIZES, PALETTE, apply_max_ticks, save_figure,
    )

    apply_paper_style()                          # rcParams set globally
    fig, ax = plt.subplots(figsize=FIGSIZES["small"])
    ax.plot(x, y, label="...")
    apply_max_ticks(ax)                          # caps each axis at 5 ticks
    save_figure(fig, "my_plot")                  # writes images/my_plot.pdf

Pass ``apply_paper_style(grid=False)`` to disable the major grid project-wide
without editing each script.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
from cycler import cycler
from matplotlib.ticker import LogLocator, MaxNLocator


# Okabe-Ito colorblind-safe palette (Wong 2011, Nat. Methods 8, 441).
PALETTE: list[str] = [
    "#000000",  # black
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#009E73",  # bluish green
    "#F0E442",  # yellow
    "#0072B2",  # blue
    "#D55E00",  # vermilion
    "#CC79A7",  # reddish purple
]

FIGSIZES: dict[str, tuple[float, float]] = {
    "small": (3.5, 3.5),
    "large": (7.0, 7.0),
}

LABEL_SIZE = 15
TITLE_SIZE = 18
LINEWIDTH = 1.5
LINEWIDTH_THIN = 1.2
MAX_TICKS = 5

PROJECT_ROOT = Path(__file__).resolve().parents[2]
IMAGES_DIR = PROJECT_ROOT / "images"


def apply_paper_style(*, grid: bool = True, usetex: bool = True) -> None:
    """Set matplotlib rcParams to the project's standard template.

    With ``usetex=True`` (default) all labels are rendered by the system LaTeX
    in Computer Modern, matching the revtex4-1 body font of ``derivation.tex``.
    Pass ``usetex=False`` to fall back to matplotlib's built-in mathtext (useful
    on machines without a full TeX install).
    """
    mpl.rcParams.update({
        "text.usetex": usetex,
        "text.latex.preamble": r"\usepackage{amsmath}\usepackage{amssymb}",
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman", "CMU Serif", "DejaVu Serif"],
        "font.size": LABEL_SIZE,
        "axes.titlesize": TITLE_SIZE,
        "axes.labelsize": LABEL_SIZE,
        "xtick.labelsize": LABEL_SIZE - 2,
        "ytick.labelsize": LABEL_SIZE - 2,
        "legend.fontsize": LABEL_SIZE - 2,
        "figure.titlesize": TITLE_SIZE,
        "lines.linewidth": LINEWIDTH,
        "axes.prop_cycle": cycler(color=PALETTE),
        "axes.grid": grid,
        "grid.alpha": 0.3,
        "grid.linestyle": ":",
        "grid.linewidth": 0.6,
        # 300 dpi keeps rasterized inserts (e.g. pcolormesh meshes) crisp inside
        # an otherwise-vector PDF.
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "figure.dpi": 110,
        # PDF backend: keep text as text (selectable, vector) rather than paths.
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def pcolormesh_rasterized(ax, *args, **kwargs):
    """``ax.pcolormesh`` with the mesh rasterized inside an otherwise-vector PDF.

    Axes, ticks, labels, and the colorbar stay vector; only the dense colour mesh
    is embedded as a bitmap at the current ``savefig.dpi``.
    """
    kwargs.setdefault("shading", "auto")
    kwargs["rasterized"] = True
    return ax.pcolormesh(*args, **kwargs)


def apply_max_ticks(ax=None, n: int = MAX_TICKS, *, axes: Iterable[str] = ("x", "y")) -> None:
    """Limit each requested axis to at most ``n`` major ticks.

    Works for both linear and log scales. Call *after* you set xscale/yscale.
    """
    ax = ax if ax is not None else plt.gca()
    axis_map = {"x": (ax.xaxis, ax.get_xscale()), "y": (ax.yaxis, ax.get_yscale())}
    for key in axes:
        axis_obj, scale = axis_map[key]
        if scale == "log":
            axis_obj.set_major_locator(LogLocator(numticks=n))
        else:
            axis_obj.set_major_locator(MaxNLocator(nbins=n - 1, prune="both"))


def save_figure(fig, name: str, *, ext: str = "pdf", subdir: str | None = None) -> Path:
    """Save ``fig`` to ``images/<subdir>/<name>.<ext>`` (creates the dir)."""
    target_dir = IMAGES_DIR if subdir is None else IMAGES_DIR / subdir
    target_dir.mkdir(parents=True, exist_ok=True)
    out = target_dir / f"{name}.{ext}"
    fig.savefig(out)
    return out
