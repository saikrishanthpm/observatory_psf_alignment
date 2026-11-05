"""
make_noise_moves_figure.py -- figure for the draft thesis section on known moves of the Hubble
secondary: (a) on-axis PSF at the nominal position and after the four known tilts of +-0.1 deg;
(b) predicted gain in precision over the same number of exposures without moving, versus the
size of the tilt moves, from noise_moves_analysis.npz, with the two simulated designs (+-0.05 and
+-0.1 deg) marked. Style from make_noise_figures.py.

    python thesis_drafts/make_noise_moves_figure.py    -> thesis_drafts/figures/noise_moves.png
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
import make_noise_figures as nf                       # noqa: E402  (fonts, inks, axes style)
import hubble_moves_pool as hm                        # noqa: E402
import matplotlib.pyplot as plt                       # noqa: E402
from optiland.psf import MMDFTPSF                     # noqa: E402

TRUTH = np.array([52.22, -113.86, 38.99, 0.13, -0.04])
MOVES = [(0, 0), (0.1, 0), (-0.1, 0), (0, 0.1), (0, -0.1)]
TITLES = ["nominal", "stx +0.1°", "stx −0.1°", "sty +0.1°", "sty −0.1°"]
BLUE = nf.COLOR["hubble"]
OUT = os.path.join(HERE, "figures", "noise_moves.png")


def psf(dtx, dty):
    t = TRUTH + np.array([0, 0, 0, dtx, dty])
    p = MMDFTPSF(hm.hubble(0.55, *t), field=(0, 0), wavelength=0.55, grid_size=64).psf[96:160, 96:160]
    return p / 100.0                                  # Optiland puts the aligned peak at 100


def main():
    d = np.load(os.path.join(HERE, "noise_moves_analysis.npz"))
    sizes, gains = d["sizes"], d["gains"]
    psfs = [np.log10(np.maximum(psf(*m), 1e-12)) for m in MOVES]
    hi = np.ceil(max(p.max() for p in psfs) * 2) / 2
    lo = hi - 3

    fig = plt.figure(figsize=(7.0, 4.0), dpi=300, facecolor="w")
    gs = fig.add_gridspec(2, 6, height_ratios=[1.0, 1.35], width_ratios=[1, 1, 1, 1, 1, 0.07],
                          left=0.08, right=0.93, top=0.93, bottom=0.115, hspace=0.3, wspace=0.08)
    for i, (p, title) in enumerate(zip(psfs, TITLES)):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(p, cmap="viridis", vmin=lo, vmax=hi, origin="lower", interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color(nf.HAIR)
            s.set_linewidth(0.6)
        ax.set_title(title, fontsize=nf.FS, color=nf.INK, pad=3)
        if i == 0:
            nf.panel_letter(ax, "a", dx=-0.04, dy=1.1)
    cax = fig.add_subplot(gs[0, 5])
    cax.imshow(np.linspace(lo, hi, 256)[:, None], cmap="viridis", vmin=lo, vmax=hi, origin="lower",
               aspect="auto", extent=[0, 1, lo, hi])
    cax.yaxis.tick_right()
    cax.set_xticks([])
    cax.tick_params(colors=nf.INK2, labelcolor=nf.INK2, labelsize=nf.FS - 1, width=0.6, length=2)
    for s in cax.spines.values():
        s.set_color(nf.HAIR)
        s.set_linewidth(0.6)
    cax.set_ylabel(r"$\log_{10}(I/I_{\mathrm{peak,\,ideal}})$", color=nf.INK2, fontsize=nf.FS - 1)
    cax.yaxis.set_label_position("right")

    ax = fig.add_subplot(gs[1, :5])
    nf.style_axes(ax)
    four = gains[:, [0, 1, 3, 4]]
    ax.axhline(1, color=nf.INK2, lw=0.7, ls=(0, (3, 2)), zorder=1)
    for x in (0.05, 0.1):                             # the two simulated designs
        ax.axvline(x, color=nf.HAIR, lw=0.8, zorder=1)
    ax.fill_between(sizes, four.min(axis=1), four.max(axis=1), color=BLUE, alpha=0.18, lw=0, zorder=2)
    ax.plot(sizes, four.mean(axis=1), color=BLUE, lw=1.5, marker="o", ms=2.5, zorder=3)
    ax.plot(sizes, gains[:, 2], color=BLUE, lw=1.5, ls=(0, (1, 1.2)), marker="o", ms=2.5, zorder=3)
    ax.set_xlim(0, 0.172)
    ax.set_ylim(0.6, 7.9)
    ax.set_xlabel("Tilt of each known move about x and about y (degrees)", color=nf.INK2)
    ax.set_ylabel("Reduction in error (×)", color=nf.INK2)
    ax.text(0.152, 6.35, "despace", color=nf.INK, ha="left", va="bottom")
    ax.text(0.118, 3.5, "decenters and tilts", color=nf.INK, ha="left", va="bottom")
    ax.text(0.169, 1.12, "no gain over five exposures without moving", color=nf.INK2, ha="right",
            va="bottom", fontsize=nf.FS - 1)
    for x in (0.05, 0.1):
        ax.text(x + 0.0015, 7.75, f"simulated run, ±{x:g}°", color=nf.INK2, ha="left", va="top", fontsize=nf.FS - 1)
    nf.panel_letter(ax, "b", dx=-0.008, dy=1.03)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=300, facecolor="w")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
