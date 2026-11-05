"""
make_tma_progression_figure.py
==============================

TMA counterpart of the Hubble PSF-progression figure
(make_psf_alignment_figure.m -> psf_progression.png), drawn in matplotlib with
the same geometry, palette, font and sizes so the two figures read as a pair.

With a logged DE run (has_trace = 1; fig_gen_de.py -> export_tma_figdata.py)
the layout is the Hubble one:

* (a) PSF strip: initial, Nautilus median, DE snapshots A-D, final.
* (b) RMS WFE of the best DE member per generation, snapshots marked,
  Marechal lambda/14 line.
* (c) every parameter vs generation (small multiples) with the injected truth.

Without a parameter log (tma_moredof2.ipynb, cell 27: has_trace = 0) the
layout is reduced: initial / Nautilus / final / aligned, the DE cost per
generation, and Nautilus start vs DE final per parameter.

    python make_tma_progression_figure.py                           # traced run
    python make_tma_progression_figure.py tma_moredof2_figdata.mat  # notebook run
    python make_tma_progression_figure.py hubble_noise_figdata.mat OUT  # any .mat, to OUT.{png,pdf}

Output: O_Mirror__Where_Art_Thou___RASTI_/figures/tma_psf_progression.{png,pdf}
(traced) or tma_psf_progression_notrace.{png,pdf} (no trace), unless an output
path (without extension) is given as the second argument.
Run with any Python that has numpy, scipy and matplotlib.
"""
import os
import sys

import numpy as np
import scipy.io as sio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, MaxNLocator, NullFormatter, NullLocator

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "tma_fig_gen_de_figdata.mat")
OUTDIR = os.path.join(HERE, "O_Mirror__Where_Art_Thou___RASTI_", "figures")
MARECHAL = False  # lambda/14 line in panel (b) of the traced layout

# ------------------------------------------------------------ style (as .m)
INK, INK2 = "#0b0b0b", "#52514e"
GRID, HAIR = "#e4e3df", "#cfcdc6"
SERIES, TRUTH = "#2a78d6", "#52514e"
FONT, FS = "Helvetica", 8


def register_helvetica_faces():
    """matplotlib reads only the first face of macOS's Helvetica.ttc (Regular),
    so bold text silently falls back to Regular. Extract Bold, Oblique and
    Bold Oblique to a temporary folder and register them."""
    import tempfile
    from matplotlib import font_manager as fm
    ttc = "/System/Library/Fonts/Helvetica.ttc"
    try:
        from fontTools.ttLib import TTCollection
    except ImportError:
        return
    if not os.path.exists(ttc):
        return
    outdir = os.path.join(tempfile.gettempdir(), "helvetica_faces")
    os.makedirs(outdir, exist_ok=True)
    for face in TTCollection(ttc).fonts:
        full = face["name"].getDebugName(4)
        if full in ("Helvetica Bold", "Helvetica Oblique", "Helvetica Bold Oblique"):
            path = os.path.join(outdir, full.replace(" ", "-") + ".ttf")
            if not os.path.exists(path):
                face.save(path)
            fm.fontManager.addfont(path)


register_helvetica_faces()
plt.rcParams.update({
    "font.family": FONT, "font.size": FS,
    "mathtext.fontset": "custom", "mathtext.rm": FONT, "mathtext.it": f"{FONT}:italic",
    "mathtext.bf": f"{FONT}:bold", "axes.unicode_minus": True,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

S = sio.loadmat(DATA, squeeze_me=True)
has_trace = float(S.get("has_trace", 0)) > 0
OUT = sys.argv[2] if len(sys.argv) > 2 else \
    os.path.join(OUTDIR, "tma_psf_progression" if has_trace else "tma_psf_progression_notrace")
names = [str(n) for n in S["param_names"]]
units = [str(u) for u in S["param_units"]]
truth = np.asarray(S["theta_true"], float)
ns = np.asarray(S["theta_ns"], float)
final = np.asarray(S["theta_final"], float)
gen = np.asarray(S["gen"], float)
cost = np.asarray(S["objective"], float)
n_gen = int(gen[-1])

if has_trace:
    wfe = np.asarray(S["wfe_trace"], float)
    theta = np.asarray(S["theta_trace"], float)
    snapG = np.atleast_1d(S["snap_gen"]).astype(int)
    K = snapG.size
    psf_snap = np.asarray(S["psf_snap"], float).reshape(*S["psf_final"].shape, K)
    psfs = [S["psf_initial"], S["psf_ns"]] + [psf_snap[:, :, k] for k in range(K)] + [S["psf_final"]]
    wfes = ([float(S["wfe_initial"]), float(S["wfe_ns"])] + list(np.atleast_1d(S["wfe_snap"]))
            + [float(S["wfe_final"])])
    ptitl = ["initial", "median"] + [chr(65 + k) for k in range(K)] + ["final"]
    pgen = ["", ""] + [f"gen {g}" for g in snapG] + [f"gen {n_gen}"]
    groups = [([0], "Start"), ([1], "Nautilus"),
              (list(range(2, len(psfs))), "Differential evolution")]
    bold = set(range(2, 2 + K))
else:
    psfs = [S["psf_initial"], S["psf_ns"], S["psf_final"], S["psf_ideal"]]
    wfes = [float(S["wfe_initial"]), float(S["wfe_ns"]), float(S["wfe_final"]), 0.0]
    ptitl = ["initial", "median", "final", "aligned"]
    pgen = ["", "", f"gen {n_gen}", ""]
    groups = [([0], "Start"), ([1], "Nautilus"), ([2], "Differential evolution"),
              ([3], "Reference")]
    bold = set()
nP, P = len(psfs), len(names)

# ------------------------------------------------------------ geometry (in)
W = 7.0
L, Rm, gapP = 0.62, 0.72, 0.05
s = min(1.3, (W - L - Rm - (nP - 1) * gapP) / nP)
hdrH, subH, belowH = 0.34, 0.15, 0.32
wfeH = 1.30
xlabH = 0.0 if has_trace else 0.40   # (b) has no x labels when (c) shares its axis
nCol = 5
nRow = int(np.ceil(P / nCol))
keyH, titleH, miniH, miniGapY, miniGapX = 0.20, 0.17, 0.66, 0.08, 0.42
parH = keyH + nRow * (titleH + miniH) + (nRow - 1) * miniGapY
botH = 0.48 if has_trace else 0.30
H = 0.06 + hdrH + subH + s + belowH + 0.30 + wfeH + xlabH + 0.14 + parH + botH
plotR = L + nP * s + (nP - 1) * gapP

fig = plt.figure(figsize=(W, H), dpi=300, facecolor="w")


def axes_in(x, y, w, h):
    return fig.add_axes([x / W, y / H, w / W, h / H])


ov = axes_in(0, 0, W, H)
ov.set_xlim(0, W)
ov.set_ylim(0, H)
ov.axis("off")


def text(x, y, t, **kw):
    kw.setdefault("color", INK)
    kw.setdefault("fontsize", FS)
    return ov.text(x, y, t, **kw)


def panel_letter(y, ch):
    text(0.06, y, f"({ch})", fontweight="bold", fontsize=FS + 1, va="top")


def style_axes(ax, fs=FS):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK2)
        ax.spines[side].set_linewidth(0.6)
    ax.tick_params(colors=INK2, labelcolor=INK2, labelsize=fs, width=0.6, length=3,
                   direction="out", which="major")
    ax.tick_params(which="minor", length=0)
    ax.grid(True, color=GRID, linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.set_facecolor("w")


def fmt3(v):
    if v == 0:
        return "0"
    if v >= 0.001:
        return f"{v:.3g}"
    e = int(np.floor(np.log10(v)))
    return rf"{v / 10 ** e:.2g}$\times10^{{{e}}}$"


def nice_ticks(lim, n):
    """The finest 1-2-5 step that puts at most n (and at least 2) ticks in lim."""
    span = lim[1] - lim[0]
    if not span > 0:
        return [lim[0]]
    mag = 10.0 ** np.floor(np.log10(span))
    for st in np.array([0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10]) * mag:
        tt = np.arange(np.ceil(lim[0] / st), np.floor(lim[1] / st) + 1) * st
        if 2 <= tt.size <= n:
            tt[np.abs(tt) < span * 1e-9] = 0
            return tt
    return [lim[0]]


def gen_axis(ax, xl, labels=True):
    ax.set_xscale("log")
    ax.set_xlim(*xl)
    ax.xaxis.set_major_locator(FixedLocator([t for t in (1, 10, 100, 1000) if t <= xl[1]]))
    if labels:
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{int(v)}"))
    else:
        ax.xaxis.set_major_formatter(NullFormatter())
    ax.xaxis.set_minor_locator(NullLocator())


def param_label(p, k):
    # Greek mu (U+03BC): Helvetica draws the micro sign (U+00B5) slanted
    unit = "μm" if units[p] == "um" else units[p]
    return f"{names[p]} ($10^{{-{k}}}$ {unit})" if k else f"{names[p]} ({unit})"


# --------------------------------------------------------- (a) PSF strip
lo = -4.0
y = H - 0.06 - hdrH - subH - s
xP = [L + i * (s + gapP) for i in range(nP)]
for i, psf in enumerate(psfs):
    ax = axes_in(xP[i], y, s, s)
    ax.imshow(np.log10(np.maximum(psf, 10 ** lo)), cmap="viridis", vmin=lo, vmax=0,
              interpolation="nearest", aspect="auto")
    ax.set_axis_off()
    text(xP[i] + s / 2, y + s + 0.05, ptitl[i], ha="center", va="baseline",
         fontweight="bold" if i in bold else "normal")
    text(xP[i] + s / 2, y - 0.04, f"{fmt3(wfes[i])} waves", ha="center", va="top",
         fontsize=FS - 1)
    if pgen[i]:
        text(xP[i] + s / 2, y - 0.18, pgen[i], ha="center", va="top", fontsize=FS - 1,
             color=INK2)
    if i == 0:
        npx = psf.shape[1]
        lamd = float(S["lamD_pix"])
        span = npx / lamd
        Lb = [c for c in (1, 2, 5, 10, 20, 50) if c <= 0.3 * span][-1]
        xb, yb2 = 0.08 * npx, 0.90 * psf.shape[0]
        ax.plot([xb, xb + Lb * lamd], [yb2, yb2], color="w", linewidth=1.5,
                solid_capstyle="butt")
        ax.text(xb + Lb * lamd / 2, yb2 - 0.04 * npx, f"{Lb}λ/D", color="w",
                ha="center", va="bottom", fontsize=FS - 1)
text(L - 0.06, y - 0.04, "WFE", ha="right", va="top", fontsize=FS - 1, color=INK2)
text(L - 0.06, y - 0.18, "DE", ha="right", va="top", fontsize=FS - 1, color=INK2)

yb = y + s + subH + 0.06
for idx, label in groups:
    x0, x1 = xP[idx[0]], xP[idx[-1]] + s
    ov.plot([x0, x0, x1, x1], [yb - 0.04, yb, yb, yb - 0.04], color=INK2, linewidth=0.6,
            solid_joinstyle="miter")
    text((x0 + x1) / 2, yb + 0.05, label, ha="center", va="baseline", fontweight="bold")
panel_letter(yb + 0.03 + 0.15, "a")

axC = axes_in(xP[-1] + s + 0.10, y, 0.08, s)
axC.imshow(np.linspace(lo, 0, 256)[:, None], cmap="viridis", vmin=lo, vmax=0,
           aspect="auto", origin="lower", extent=[0, 1, lo, 0])
axC.set_xticks([])
axC.yaxis.tick_right()
axC.set_yticks(np.arange(lo, 0.5, 1))
axC.set_yticklabels([rf"$10^{{{int(e)}}}$" for e in np.arange(lo, 0.5, 1)])
for side in ("top", "bottom", "left"):
    axC.spines[side].set_visible(False)
axC.spines["right"].set_color(INK2)
axC.spines["right"].set_linewidth(0.6)
axC.tick_params(colors=INK2, labelcolor=INK2, labelsize=FS - 1, width=0.6, length=3)
axC.yaxis.set_label_position("right")
axC.set_ylabel(r"I / I$_\mathrm{peak,\ ideal}$", fontsize=FS - 1, color=INK, labelpad=4)

# ------------------------------------------- (b) RMS WFE (or DE cost) per generation
yW = y - belowH - 0.30 - wfeH
axW = axes_in(L, yW, plotR - L, wfeH)
style_axes(axW)
xl = (1, max(n_gen, 2))
curve = wfe if has_trace else cost
yl = (curve[curve > 0].min() / 2, curve.max() * 2)
if has_trace and MARECHAL:
    yl = (min(yl[0], 1 / 14 / 2), max(yl[1], 1 / 14 * 2))
gen_axis(axW, xl, labels=not has_trace)
axW.set_yscale("log")
axW.set_ylim(*yl)
axW.yaxis.set_minor_locator(NullLocator())
if has_trace:
    for g in snapG:
        axW.plot([g, g], yl, color=HAIR, linewidth=0.5, zorder=1)
    if MARECHAL:
        axW.plot(xl, [1 / 14] * 2, color=INK2, linewidth=0.6, zorder=2)
        # label under the line at the right end, where a converged curve sits far below
        axW.text(xl[1], 1 / 14 / 1.2, "Maréchal λ/14 ", ha="right", va="top",
                 fontsize=FS - 1, color=INK2, zorder=2,
                 bbox=dict(facecolor="w", edgecolor="none", pad=1))
axW.step(gen, curve, where="post", color=SERIES, linewidth=1.2, zorder=3)
if has_trace:
    # letters sit on the top edge; the hairline below each one leads to its dot.
    # Letters closer than ~2% of the axis width are nudged apart.
    fwd = lambda x: (np.log10(x) - np.log10(xl[0])) / (np.log10(xl[1]) - np.log10(xl[0]))
    inv = lambda u: 10 ** (np.log10(xl[0]) + u * (np.log10(xl[1]) - np.log10(xl[0])))
    u = fwd(snapG.astype(float))
    for k in range(1, K):
        u[k] = max(u[k], u[k - 1] + 0.022)
    for k, g in enumerate(snapG):
        axW.plot(g, wfe[g - 1], "o", markersize=5, markerfacecolor=INK, markeredgecolor="w",
                 markeredgewidth=0.8, zorder=4)
        axW.text(inv(u[k]), yl[1], chr(65 + k), ha="center", va="bottom", fontsize=FS,
                 fontweight="bold", color=INK, clip_on=False)
axW.plot(gen[-1], curve[-1], "o", markersize=5, markerfacecolor=INK, markeredgecolor="w",
         markeredgewidth=0.8, zorder=4, clip_on=False)
axW.text(xl[1], curve[-1], "  final", ha="left", va="center", fontsize=FS - 1, color=INK,
         clip_on=False)
if has_trace:
    axW.set_ylabel("RMS WFE (waves)", fontsize=FS, color=INK)
else:
    axW.set_ylabel("DE cost", fontsize=FS, color=INK)
    axW.set_xlabel("DE generation", fontsize=FS, color=INK)
    text(plotR, yW + wfeH + 0.04, str(S["cost_label"]), ha="right", va="bottom",
         fontsize=FS - 1, color=INK2)
panel_letter(yW + wfeH + 0.14, "b")

# ------------------------------------------------------------ (c) parameters
# Tilts are shown in scaled units so the tick labels stay short; tilts of a
# tenth of a degree or more (Hubble) read fine in degrees.
scale = {"ptx": 3, "pty": 3, "stx": 3, "sty": 3} if has_trace else \
        {"ptx": 4, "pty": 4, "stx": 3, "sty": 3}
if np.abs(truth[[u == "deg" for u in units]]).max(initial=0) >= 0.1:
    scale = {}
yTop = yW - xlabH - 0.14
ky = yTop - 0.08
kx = L
kx2 = kx + 1.35
if has_trace:
    ov.plot([kx, kx + 0.22], [ky, ky], color=SERIES, linewidth=1.2)
    text(kx + 0.27, ky, "DE best estimate", va="center", fontsize=FS - 1)
    ov.plot([kx2, kx2 + 0.22], [ky, ky], color=TRUTH, linewidth=0.8)
    text(kx2 + 0.27, ky, "injected truth", va="center", fontsize=FS - 1)
else:
    ov.plot(kx + 0.06, ky, "o", markersize=5, markerfacecolor="w", markeredgecolor=INK2,
            markeredgewidth=0.9)
    text(kx + 0.16, ky, "Nautilus start", va="center", fontsize=FS - 1)
    ov.plot(kx2 + 0.06, ky, "o", markersize=5, markerfacecolor=SERIES, markeredgecolor="w",
            markeredgewidth=0.8)
    text(kx2 + 0.16, ky, "DE final", va="center", fontsize=FS - 1)
    kx3 = kx2 + 1.35
    ov.plot([kx3, kx3 + 0.22], [ky, ky], color=TRUTH, linewidth=0.8)
    text(kx3 + 0.27, ky, "injected truth", va="center", fontsize=FS - 1)
panel_letter(yTop, "c")

miniW = (plotR - L - (nCol - 1) * miniGapX) / nCol
for p in range(P):
    r, c = divmod(p, nCol)
    xm = L + c * (miniW + miniGapX)
    ym = yTop - keyH - (r + 1) * (titleH + miniH) - r * miniGapY
    ax = axes_in(xm, ym, miniW, miniH)
    style_axes(ax, FS - 1)
    k = scale.get(names[p], 0)
    f = 10.0 ** k
    if has_trace:
        v, v_t = theta[:, p] * f, truth[p] * f
        lim = (min(v.min(), v_t), max(v.max(), v_t))
        pad = 0.08 * max(lim[1] - lim[0], 1e-12)
        ylim = (lim[0] - pad, lim[1] + pad)
        ax.set_ylim(*ylim)
        ax.set_yticks(nice_ticks(ylim, 4))
        bottom_row = not (r < nRow - 1 and p + nCol < P)
        gen_axis(ax, xl, labels=bottom_row)
        for g in snapG:
            ax.plot([g, g], ylim, color=HAIR, linewidth=0.5, zorder=1)
        ax.plot(xl, [v_t, v_t], color=TRUTH, linewidth=0.8, zorder=2)
        ax.step(gen, v, where="post", color=SERIES, linewidth=1.0, zorder=3)
        if bottom_row:
            ax.set_xlabel("DE generation", fontsize=FS - 1, color=INK)
    else:
        v_t, v_ns, v_de = truth[p] * f, ns[p] * f, final[p] * f
        lim = (min(v_t, v_ns, v_de), max(v_t, v_ns, v_de))
        pad = 0.15 * max(lim[1] - lim[0], 1e-12)
        ax.set_ylim(lim[0] - pad, lim[1] + pad)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=3, steps=[1, 2, 2.5, 5, 10]))
        ax.set_xlim(-0.45, 1.45)
        ax.set_xticks([0, 1])
        ax.grid(False, axis="x")
        ax.axhline(v_t, color=TRUTH, linewidth=0.8, zorder=2)
        ax.plot([0, 1], [v_ns, v_de], color=HAIR, linewidth=1.5, zorder=3)
        ax.plot(0, v_ns, "o", markersize=5, markerfacecolor="w", markeredgecolor=INK2,
                markeredgewidth=0.9, zorder=4, clip_on=False)
        ax.plot(1, v_de, "o", markersize=5, markerfacecolor=SERIES, markeredgecolor="w",
                markeredgewidth=0.8, zorder=5, clip_on=False)
        if r == nRow - 1:
            ax.set_xticklabels(["Nautilus", "DE"])
        else:
            ax.xaxis.set_major_formatter(NullFormatter())
    text(xm, ym + miniH + 0.03, param_label(p, k), ha="left", va="bottom", fontsize=FS - 1)

os.makedirs(os.path.dirname(os.path.abspath(OUT)), exist_ok=True)
fig.savefig(OUT + ".png", dpi=300, facecolor="w")
fig.savefig(OUT + ".pdf", facecolor="w")
print(f"wrote {OUT}.png and .pdf  ({W:.2f} x {H:.2f} in)")

if has_trace:
    g = ", ".join(str(v) for v in snapG)
    print(
        "\nSuggested caption:\n"
        f"(a) On-axis PSFs at lambda = {float(S['wavelength_um']):.2f} um: the observed misaligned PSF "
        f"(Gaussian noise, sigma = {float(S['noise_sigma']):g}% of the ideal peak), after Nautilus "
        f"(posterior median), during differential evolution (DE) at generations {g} "
        f"(A-{chr(64 + K)}), and after {n_gen} generations (final). Intensity is relative to the "
        "aligned system's peak, so each panel's peak equals its Strehl ratio (log stretch, floor "
        "1e-4). Values are RMS wavefront error in waves, the RMS of the OPD difference from the "
        "nominal design. (b) RMS WFE of the best DE member at each generation; letters mark the "
        "PSFs in (a). " + ("Gray line: Marechal criterion, lambda/14. " if MARECHAL else "") +
        "(c) Best-member value of each "
        "alignment parameter at each generation (blue) and the injected truth (gray).\n")
