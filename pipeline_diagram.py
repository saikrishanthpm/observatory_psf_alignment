"""
pipeline_diagram.py - simple block diagram of the PSF-based alignment pipeline
(hubbletilus.ipynb / tma_moredof2.ipynb).

    python pipeline_diagram.py      -> pipeline_diagram.pdf / .svg / .png

Box wording is in MAIN and BRANCH below.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = "pipeline_diagram"
FONT = "Liberation Sans"            # metric-compatible with Arial / Helvetica
plt.rcParams.update({"font.family": FONT, "svg.fonttype": "none", "pdf.fonttype": 42})

INK, INK2 = "#0b0b0b", "#52514e"
GRAY_FILL, GRAY_EDGE = "#f3f2ef", "#8a8983"
BLUE_FILL, BLUE_EDGE = "#dbe8f9", "#2a78d6"
FS = 7.0
HEAD = "-|>,head_length=3.4,head_width=2.0"

# main flow, left to right
MAIN = [
    "Simulate PSFs\nwith optiland at\neach field point",
    "Project PSFs\nonto basis\nmodes",
    "Build Gaussian\nlog-likelihood",
    "Run Nautilus\nnested sampler",
    "Posterior median\nand 2.1–97.9%\nbounds",
    "Run differential\nevolution",
    "Final estimate\nand residual\nWFE check",
]
# covariance calibration, done once before sampling (sits under boxes 2-3)
BRANCH = [
    "Repeat for\n10,000 random\nmisalignments",
    "Covariance\nmatrix for each\nfield point",
]

W = 7.0
bw, bh = 0.86, 0.62               # box size (in)
gap = (W - 0.10 - len(MAIN) * bw) / (len(MAIN) - 1)
legendH = 0.26
H = 0.08 + bh + 0.34 + bh + legendH

fig = plt.figure(figsize=(W, H))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.axis("off")


def box(x, y, text, fill, edge):
    ax.add_patch(FancyBboxPatch((x, y), bw, bh, boxstyle="round,pad=0,rounding_size=0.06",
                                fc=fill, ec=edge, lw=0.8, zorder=2))
    ax.text(x + bw / 2, y + bh / 2, text, fontsize=FS, color=INK, ha="center", va="center",
            linespacing=1.25, zorder=3)


def arrow(p0, p1):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=HEAD, color=INK2, lw=0.8,
                                 shrinkA=0, shrinkB=0, zorder=4))


topY = H - 0.08 - bh
xs = [0.05 + i * (bw + gap) for i in range(len(MAIN))]
for x, t in zip(xs, MAIN):
    box(x, topY, t, GRAY_FILL, GRAY_EDGE)
for a, b in zip(xs[:-1], xs[1:]):
    arrow((a + bw + 0.02, topY + bh / 2), (b - 0.02, topY + bh / 2))

botY = topY - 0.34 - bh
bx = [xs[1], xs[2]]
for x, t in zip(bx, BRANCH):
    box(x, botY, t, BLUE_FILL, BLUE_EDGE)
arrow((xs[1] + bw / 2, topY - 0.02), (xs[1] + bw / 2, botY + bh + 0.02))     # project -> repeat
arrow((bx[0] + bw + 0.02, botY + bh / 2), (bx[1] - 0.02, botY + bh / 2))     # repeat -> covariance
arrow((xs[2] + bw / 2, botY + bh + 0.02), (xs[2] + bw / 2, topY - 0.02))     # covariance -> likelihood

# legend
ly = botY - legendH / 2 - 0.02
lx = xs[3]
ax.add_patch(FancyBboxPatch((lx, ly - 0.05), 0.16, 0.10, boxstyle="round,pad=0,rounding_size=0.02",
                            fc=BLUE_FILL, ec=BLUE_EDGE, lw=0.8))
ax.text(lx + 0.22, ly, "computed once, before sampling", fontsize=FS - 1, color=INK2, va="center")

for ext in ("pdf", "svg"):
    fig.savefig(f"{OUT}.{ext}")
fig.savefig(f"{OUT}.png", dpi=300)
print(f"wrote {OUT}.pdf/.svg/.png  ({W:.2f} x {H:.2f} in)")
