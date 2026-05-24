#!/usr/bin/env python3
r"""Pixel-digitize the two solid curves of Roper Pol et al. (2020), Fig. 1.

Real pixel extraction (NOT eyeballed) from the published figure image:
Omega_M(k)/k (fluid, upper) and Omega_GW(k)/k (GW, lower), in their convention.

Axis calibration (from frame + tick detection):
  plot box  x in [176, 855] px,  y in [31, 444] px
  x:  log10(k)        = 2  + (x - 176)/244     (major ticks 10^2,10^3,10^4)
  y:  log10(Omega/k)  = -5 - (y -  31)/29.5    (top frame 10^-5, 14 decades to 10^-19)

The two solid curves are well separated in y; for each column we take the centre of
the longest contiguous dark run in the upper and lower bands, then despike.
Output: Notebooks/roperpol_fig1_{fluid,gw}.csv and an overlay PNG to verify.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

IMG = Path("/home/mgurgeni/.claude/image-cache/5d98e7c8-0c54-4985-833d-b12f0c43513c/1.png")
OUT = Path(__file__).resolve().parent
L, R, T, B = 176, 855, 31, 444
PER_DEC_X, PER_DEC_Y = 244.0, 29.5


def px_to_data(x, y):
    k = 10.0 ** (2.0 + (x - L) / PER_DEC_X)
    o = 10.0 ** (-5.0 - (y - T) / PER_DEC_Y)
    return k, o


def _longest_run_center(mask_col, y0, y1):
    """Center y of the longest contiguous True run within [y0,y1); None if none."""
    best_len = 0
    best_c = None
    run_start = None
    for y in range(y0, y1 + 1):
        on = bool(mask_col[y]) if y < len(mask_col) else False
        if on and run_start is None:
            run_start = y
        if (not on or y == y1) and run_start is not None:
            end = y if not on else y + 1
            if end - run_start > best_len:
                best_len = end - run_start
                best_c = 0.5 * (run_start + end - 1)
            run_start = None
    return best_c, best_len


def _despike(xs, ys, win=22, tol=0.22, passes=4):
    """Iteratively drop points whose log-y deviates from a rolling median by > tol decades.

    Wide window + repeats so localized contamination (in-plot text glyphs ~10-15 px wide,
    dash-dot guide segments) is rejected while the continuous curve survives.
    """
    xs = np.asarray(xs, float)
    ys = np.asarray(ys, float)
    for _ in range(passes):
        ly = np.log10(ys)
        keep = np.ones(len(ys), bool)
        for i in range(len(ys)):
            lo, hi = max(0, i - win), min(len(ys), i + win + 1)
            med = np.median(np.concatenate([ly[lo:i], ly[i + 1:hi]]))  # exclude self
            if abs(ly[i] - med) > tol:
                keep[i] = False
        if keep.all():
            break
        xs, ys = xs[keep], ys[keep]
    return xs, ys


def digitize():
    a = np.array(Image.open(IMG).convert("RGB")).mean(2)
    dark = a < 120
    # upper band = fluid, lower band = GW (clean gap at y~210)
    bands = {"fluid": (33, 180), "gw": (220, 443)}  # fluid y1<198 avoids the 10^-11 label
    out = {}
    for name, (y0, y1) in bands.items():
        ks, os = [], []
        for x in range(L + 1, R):
            c, ln = _longest_run_center(dark[:, x], y0, y1)
            if c is None or ln < 2 or ln > 14:   # reject specks and thick text blobs
                continue
            k, o = px_to_data(x, c)
            ks.append(k)
            os.append(o)
        ks, os = _despike(ks, os)
        out[name] = (np.array(ks), np.array(os))
        np.savetxt(OUT / f"roperpol_fig1_{name}.csv",
                   np.column_stack(out[name]), delimiter=",",
                   header="k,Omega_over_k", comments="")
        print(f"{name}: {len(ks)} points, "
              f"k in [{ks.min():.3g},{ks.max():.3g}], "
              f"Omega/k in [{os.min():.2e},{os.max():.2e}]")
    return out


def overlay(out):
    """Save a PNG with digitized points over the original to verify the extraction."""
    import matplotlib.pyplot as plt

    img = np.array(Image.open(IMG).convert("RGB"))
    fig, ax = plt.subplots(figsize=(8.7, 5.35), dpi=110)
    ax.imshow(img)
    for name, color in (("fluid", "red"), ("gw", "deepskyblue")):
        ks, os = out[name]
        xs = L + np.log10(ks / 100.0) * PER_DEC_X
        ys = T - (np.log10(os) + 5.0) * PER_DEC_Y
        ax.plot(xs, ys, ".", color=color, ms=2.4, label=name)
    ax.legend(loc="upper right")
    ax.set_title("digitized points over original (verify)")
    fig.tight_layout()
    p = OUT / "roperpol_fig1_overlay.png"
    fig.savefig(p, dpi=110)
    plt.close(fig)
    print(f"wrote {p}")


if __name__ == "__main__":
    out = digitize()
    overlay(out)
