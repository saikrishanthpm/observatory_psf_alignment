"""
make_noise_figures.py -- thesis figures drawn from existing data (no new runs).

1. Residual OPD maps (initial / Nautilus median / DE final) for each telescope,
   from hubble_noise_figdata.mat and tma_fig_gen_de_figdata.mat
   -> Masters_Thesis/figures/hubble_opd.png, tma_opd.png
2. Parameter errors and on-axis WFE versus noise level, from noise_analysis.npz
   -> Masters_Thesis/figures/noise_sweep.png
3. Noise-limited error ellipses along the strongest degeneracies, with the run
   residuals, from noise_analysis.npz (covariances) and the two figdata files
   (residuals) -> Masters_Thesis/figures/noise_ellipses.png

Style (font, inks, series blue) matches make_tma_progression_figure.py. Colour
follows the telescope throughout: blue for Hubble, orange for the TMA.

    python make_noise_figures.py
"""
import os
import tempfile

import numpy as np
import scipy.io as sio
from scipy.stats import chi2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, to_hex, to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FixedLocator, NullFormatter, NullLocator

HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(HERE, "Masters_Thesis", "figures")
NPZ = os.path.join(HERE, "noise_analysis.npz")
DATA = {"hubble": os.path.join(HERE, "hubble_noise_figdata.mat"),
        "tma": os.path.join(HERE, "tma_fig_gen_de_figdata.mat")}
N_REAL = {"hubble": 50, "tma": 30}       # realizations per level in noise_analysis.py
NAMES = {"hubble": ["sdx", "sdy", "sdz", "stx", "sty"],
         "tma": ["pdx", "pdy", "pdz", "sdx", "sdy", "sdz", "ptx", "pty", "stx", "sty"]}

# ------------------------------------------------------------ style (as .m)
INK, INK2 = "#0b0b0b", "#52514e"
GRID, HAIR = "#e4e3df", "#cfcdc6"
COLOR = {"hubble": "#2a78d6", "tma": "#eb6834"}     # palette slots 1 and 2
LABEL = {"hubble": "Hubble", "tma": "TMA"}
FONT, FS = "Helvetica", 8


def register_helvetica_faces():
    """matplotlib reads only the first face of macOS's Helvetica.ttc; register
    Bold, Oblique and Bold Oblique too (as in make_tma_progression_figure.py)."""
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


def panel_letter(ax, ch, dx=-0.02, dy=1.06):
    ax.text(dx, dy, f"({ch})", transform=ax.transAxes, fontweight="bold",
            fontsize=FS + 1, color=INK, ha="right", va="bottom")


def fmt_num(v):
    if v == 0:
        return "0"
    if abs(v) >= 0.001:
        return f"{v:g}".replace("-", "−")
    e = int(np.floor(np.log10(abs(v))))
    return rf"{v / 10 ** e:g}$\times10^{{{e}}}$".replace("-", "−", 1)


def nice_floor(x):
    """Largest 1-2-5 number not above x."""
    e = 10.0 ** np.floor(np.log10(x))
    return max(s * e for s in (1, 2, 5) if s * e <= x)


# ------------------------------------------- diverging map: blue - gray - red
def _oklab(hex_):
    rgb = np.array(to_rgb(hex_))
    lin = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    lms = np.array([[0.4122214708, 0.5363325363, 0.0514459929],
                    [0.2119034982, 0.6806995451, 0.1073969566],
                    [0.0883024619, 0.2817188376, 0.6299787005]]) @ lin
    return np.array([[0.2104542553, 0.7936177850, -0.0040720468],
                     [1.9779984951, -2.4285922050, 0.4505937099],
                     [0.0259040371, 0.7827717662, -0.8086757660]]) @ np.cbrt(lms)


def _from_oklab(lab):
    lms = np.array([[1, 0.3963377774, 0.2158037573],
                    [1, -0.1055613458, -0.0638541728],
                    [1, -0.0894841775, -1.2914855480]]) @ lab
    lin = np.array([[4.0767416621, -3.3077115913, 0.2309699292],
                    [-1.2684380046, 2.6097574011, -0.3413193965],
                    [-0.0041960863, -0.7034186147, 1.7076147010]]) @ lms ** 3
    return np.where(lin <= 0.0031308, 12.92 * lin, 1.055 * np.abs(lin) ** (1 / 2.4) - 0.055)


def diverging_cmap():
    """Blue arm from the palette's blue ramp; the red arm matches it step for
    step in OKLab lightness and relative chroma, on the hue of palette red
    #e34948; neutral gray midpoint #f0efec."""
    blue = ["#184f95", "#2a78d6", "#9ec5f4"]
    ref_b, ref_r = _oklab("#2a78d6"), _oklab("#e34948")
    hue_r = np.arctan2(ref_r[2], ref_r[1])
    c_ratio = np.hypot(*ref_r[1:]) / np.hypot(*ref_b[1:])
    red = []
    for h in blue:
        L, a, b = _oklab(h)
        C = np.hypot(a, b) * c_ratio
        for _ in range(40):                      # pull chroma in until in gamut
            rgb = _from_oklab(np.array([L, C * np.cos(hue_r), C * np.sin(hue_r)]))
            if np.all((rgb >= -1e-6) & (rgb <= 1 + 1e-6)):
                break
            C *= 0.95
        red.append(to_hex(np.clip(rgb, 0, 1)))
    cmap = LinearSegmentedColormap.from_list("div", blue + ["#f0efec"] + red[::-1])
    cmap.set_bad("w")
    return cmap


# ------------------------------------------------------------ 1. OPD maps
def opd_figure(key, out):
    S = sio.loadmat(DATA[key], squeeze_me=True)
    n_gen = int(np.asarray(S["gen"])[-1])
    maps = [S["opd_initial"], S["opd_ns"], S["opd_final"]]
    wfes = [float(S["wfe_initial"]), float(S["wfe_ns"]), float(S["wfe_final"])]
    heads = ["Start", "Nautilus", "Differential evolution"]
    titles = ["initial", "median", f"final (gen {n_gen})"]
    cmap = diverging_cmap()

    W, s, cbw, gap, L0, B = 7.0, 1.5, 0.07, 0.72, 0.2, 0.28
    H = 0.42 + s + B
    fig = plt.figure(figsize=(W, H), dpi=300, facecolor="w")
    for i, (m, w) in enumerate(zip(maps, wfes)):
        m = m - np.nanmean(m)                     # piston removed
        lim = float(np.nanmax(np.abs(m)))
        x0 = L0 + i * (s + 0.06 + cbw + gap)
        ax = fig.add_axes([x0 / W, B / H, s / W, s / H])
        ax.imshow(m, cmap=cmap, vmin=-lim, vmax=lim, origin="lower",
                  interpolation="nearest")
        ax.set_axis_off()
        ax.text(0.5, 1.03, titles[i], transform=ax.transAxes, ha="center", va="bottom",
                color=INK, fontsize=FS)
        ax.text(0.5, 1.15, heads[i], transform=ax.transAxes, ha="center", va="bottom",
                color=INK, fontsize=FS, fontweight="bold")
        ax.text(0.5, -0.04, f"RMS {fmt_num(float(f'{w:.3g}'))} waves", transform=ax.transAxes,
                ha="center", va="top", color=INK, fontsize=FS - 1)
        cax = fig.add_axes([(x0 + s + 0.06) / W, B / H, cbw / W, s / H])
        cax.imshow(np.linspace(-lim, lim, 256)[:, None], cmap=cmap, vmin=-lim, vmax=lim,
                   aspect="auto", origin="lower", extent=[0, 1, -lim, lim])
        t = nice_floor(lim)
        cax.set_xticks([])
        cax.yaxis.tick_right()
        cax.set_yticks([-t, 0, t])
        cax.set_yticklabels([fmt_num(-t), "0", fmt_num(t)])
        for side in ("top", "bottom", "left"):
            cax.spines[side].set_visible(False)
        cax.spines["right"].set_color(INK2)
        cax.spines["right"].set_linewidth(0.6)
        cax.tick_params(colors=INK2, labelcolor=INK2, labelsize=FS - 1, width=0.6, length=3)
        cax.text(0.5, 1.03, "waves", transform=cax.transAxes, ha="center", va="bottom",
                 color=INK2, fontsize=FS - 1)
    fig.savefig(out, dpi=300, facecolor="w")
    plt.close(fig)
    print(f"wrote {out}")


# ------------------------------------------------------------ 2. noise sweep
def sweep_figure(d, out):
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.55), dpi=300, facecolor="w")
    fig.subplots_adjust(left=0.08, right=0.97, bottom=0.2, top=0.8, wspace=0.42)
    xt = [1e-4, 3e-4, 1e-3, 3e-3]
    xtl = [r"$10^{-4}$", r"$3\times10^{-4}$", r"$10^{-3}$", r"$3\times10^{-3}$"]
    xx = np.array([1e-4, 3e-3])
    markers = ["o", "s", "^"]

    def x_axis(ax):
        ax.set_xscale("log")
        ax.set_xlim(6.5e-5, 4.6e-3)
        ax.xaxis.set_major_locator(FixedLocator(xt))
        ax.set_xticklabels(xtl)
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_xlabel(r"noise $\sigma$ / ideal peak", fontsize=FS, color=INK)

    def de_overlap(ys, sep=0.16):
        """Spread label positions (log10) so neighbours are at least sep apart."""
        order = np.argsort(ys)
        y = np.log10(np.asarray(ys, float))[order]
        for i in range(1, len(y)):
            y[i] = max(y[i], y[i - 1] + sep)
        out = np.empty_like(y)
        out[order] = 10 ** y
        return out

    for ax, key, ch in zip(axes[:2], ("hubble", "tma"), "ab"):
        style_axes(ax)
        x_axis(ax)
        ax.set_yscale("log")
        ax.yaxis.set_minor_formatter(NullFormatter())
        names = NAMES[key]
        sel = [names.index(n) for n in ("sdx", "sdy", "sdz")]
        s = d[f"{key}_sweep_sigma"] / 100.0                # sigma / ideal peak
        rms, lin = d[f"{key}_sweep_rms_err"], d[f"{key}_sweep_lin_err"]
        col = COLOR[key]
        ends = []
        for j, mk, dodge in zip(sel, markers, (0.93, 1.07, 1.0)):   # sdx and sdy overlap
            slope = lin[0, j] / s[0]
            ax.plot(xx, slope * xx, color=col, linewidth=0.9, alpha=0.55, zorder=2)
            err = rms[:, j] / np.sqrt(2 * N_REAL[key])
            ax.errorbar(s * dodge, rms[:, j], yerr=err, fmt=mk, markersize=4.5, color=col,
                        markerfacecolor=col, markeredgecolor="w", markeredgewidth=0.6,
                        elinewidth=0.8, capsize=0, zorder=3)
            ends.append(rms[-1, j])
        for j, y in zip(sel, de_overlap(ends)):
            ax.text(s[-1] * 1.18, y, names[j], color=INK, fontsize=FS - 1, va="center")
        ax.set_ylabel(r"RMS error ($\mu$m)", fontsize=FS, color=INK)
        ax.set_title(LABEL[key], fontsize=FS, color=INK, fontweight="bold", pad=4)
        panel_letter(ax, ch, dx=-0.12)

    ax = axes[2]
    style_axes(ax)
    x_axis(ax)
    ax.set_yscale("log")
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.axhline(1 / 14, color=INK2, linewidth=0.7, zorder=1)
    ax.text(7.2e-5, 1 / 14 * 1.12, r"Mar$\acute{\rm e}$chal $\lambda/14$", color=INK2,
            fontsize=FS - 1, va="bottom")
    ends = []
    for key, mk in (("hubble", "o"), ("tma", "s")):
        s = d[f"{key}_sweep_sigma"] / 100.0
        w, wl = d[f"{key}_sweep_wfe_median"], d[f"{key}_sweep_wfe_lin_median"]
        ax.plot(xx, wl[0] / s[0] * xx, color=COLOR[key], linewidth=0.9, alpha=0.55, zorder=2)
        ax.plot(s, w, mk, markersize=4.5, color=COLOR[key], markeredgecolor="w",
                markeredgewidth=0.6, zorder=3)
        ends.append(w[-1])
    for key, y in zip(("hubble", "tma"), de_overlap(ends)):
        ax.text(3e-3 * 1.18, y, LABEL[key], color=INK, fontsize=FS - 1, va="center")
    ax.set_ylabel("median RMS WFE (waves)", fontsize=FS, color=INK)
    ax.set_title("On-axis WFE", fontsize=FS, color=INK, fontweight="bold", pad=4)
    panel_letter(ax, "c", dx=-0.12)

    handles = [Line2D([], [], marker="o", linestyle="none", color=INK2, markersize=4.5,
                      markeredgecolor="w", markeredgewidth=0.6),
               Line2D([], [], color=INK2, linewidth=0.9, alpha=0.55)]
    fig.legend(handles, ["simulated least-squares solutions (RMS, $\\pm1\\sigma$)",
                         "linear prediction"],
               loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False,
               fontsize=FS - 1, handlelength=1.8, columnspacing=2.0, labelcolor=INK)
    fig.savefig(out, dpi=300, facecolor="w")
    plt.close(fig)
    print(f"wrote {out}")


# ------------------------------------------------------------ 3. ellipses
def ellipse_figure(d, out):
    pairs = [("hubble", "sdx", "sty"), ("hubble", "sdy", "stx"),
             ("tma", "sdx", "pty"), ("tma", "pdz", "sdz")]
    r68, r95 = np.sqrt(chi2.ppf([0.68, 0.95], 2))
    t = np.linspace(0, 2 * np.pi, 400)
    circle = np.array([np.cos(t), np.sin(t)])

    resid = {}
    for key in ("hubble", "tma"):
        S = sio.loadmat(DATA[key], squeeze_me=True)
        resid[key] = np.asarray(S["theta_true"], float) - np.asarray(S["theta_final"], float)

    fig, axes = plt.subplots(2, 2, figsize=(5.6, 4.9), dpi=300, facecolor="w")
    fig.subplots_adjust(left=0.13, right=0.97, bottom=0.1, top=0.88, wspace=0.38, hspace=0.5)
    for ax, (key, px, py), ch in zip(axes.ravel(), pairs, "abcd"):
        style_axes(ax)
        names = NAMES[key]
        i, j = names.index(px), names.index(py)
        cov = d[f"{key}_cov"]
        f = np.array([1e3 if names[k][1] == "t" else 1.0 for k in (i, j)])   # tilts in 1e-3 deg
        C2 = cov[np.ix_([i, j], [i, j])] * np.outer(f, f)
        w, V = np.linalg.eigh(C2)
        A = V @ np.diag(np.sqrt(np.clip(w, 0, None)))
        col = COLOR[key]
        e68, e95 = A @ circle * r68, A @ circle * r95
        ax.fill(*e68, color=col, alpha=0.22, linewidth=0, zorder=2)
        ax.plot(*e68, color=col, linewidth=1.0, zorder=3)
        ax.plot(*e95, color=col, linewidth=0.7, zorder=3)
        ax.axhline(0, color=HAIR, linewidth=0.6, zorder=1)
        ax.axvline(0, color=HAIR, linewidth=0.6, zorder=1)
        r = resid[key][[i, j]] * f
        ax.plot(*r, "o", markersize=5, color=INK, markeredgecolor="w", markeredgewidth=0.8,
                zorder=4)
        sd = np.sqrt(np.diag(C2))
        lim = 1.15 * np.maximum(r95 * sd, np.abs(r))
        ax.set_xlim(-lim[0], lim[0])
        ax.set_ylim(-lim[1], lim[1])
        ax.text(r[0] + 0.05 * lim[0], r[1] + 0.06 * lim[1], "run", color=INK,
                fontsize=FS - 1, va="bottom", ha="left", zorder=5)
        unit = lambda n: r"$10^{-3}$ deg" if n[1] == "t" else r"$\mu$m"
        ax.set_xlabel(f"{px} ({unit(px)})", fontsize=FS, color=INK)
        ax.set_ylabel(f"{py} ({unit(py)})", fontsize=FS, color=INK)
        rho = cov[i, j] / np.sqrt(cov[i, i] * cov[j, j])
        cond = 1 / np.sqrt(np.diag(np.linalg.pinv(cov)))[i] * f[0]
        ax.set_title(f"{LABEL[key]}: {px} and {py}", fontsize=FS, color=INK,
                     fontweight="bold", pad=4)
        note = (f"$r$ = {rho:+.3f}".replace("-", "−") + "\n"
                f"$\\sigma_{{{px}}}$ = {sd[0]:.3g} {unit(px)}\n"
                f"others known: {cond:.2g} {unit(px)}")
        # the corner away from the ellipse's long axis
        nx, ha = (0.03, "left") if rho > 0 else (0.97, "right")
        ax.text(nx, 0.97, note, transform=ax.transAxes, fontsize=FS - 1.5, color=INK2,
                va="top", ha=ha, linespacing=1.3, zorder=6)
        panel_letter(ax, ch, dx=-0.16)

    handles = [Patch(facecolor=INK2, alpha=0.22, edgecolor=INK2, linewidth=1.0),
               Line2D([], [], color=INK2, linewidth=0.7),
               Line2D([], [], marker="o", linestyle="none", color=INK, markersize=5,
                      markeredgecolor="w", markeredgewidth=0.8)]
    fig.legend(handles, ["68% region", "95% region", "run residual (truth $-$ estimate)"],
               loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False,
               fontsize=FS - 1, handlelength=1.6, columnspacing=1.6, labelcolor=INK)
    fig.savefig(out, dpi=300, facecolor="w")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    os.makedirs(OUTDIR, exist_ok=True)
    opd_figure("hubble", os.path.join(OUTDIR, "hubble_opd.png"))
    opd_figure("tma", os.path.join(OUTDIR, "tma_opd.png"))
    d = np.load(NPZ)
    sweep_figure(d, os.path.join(OUTDIR, "noise_sweep.png"))
    ellipse_figure(d, os.path.join(OUTDIR, "noise_ellipses.png"))
