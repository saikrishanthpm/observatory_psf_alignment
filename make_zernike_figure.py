"""
make_zernike_figure.py -- the first 15 Zernike modes as a pyramid, matching the
circular-Bessel basis figure (Masters_Thesis/figures/circular_bessel_basis.png).

Rows are radial orders n = 0..4. Within each row the modes are placed exactly
where the circular-Bessel figure has the function with the same n and the same
angular dependence, so each Zernike mode sits where its Fourier transform does.
The geometry (960 x 720 canvas, 138 px panels on a 144 px pitch) is copied from
that figure so the two print at the same scale.

The modes are POPPY's Noll-ordered Zernikes on the 256 x 256 grid used for the
TMA, zero outside the unit disk (as in decompose_psf, outside=0.0); each panel
has its own symmetric color scale, so zero is the same color in every panel.

    python make_zernike_figure.py
"""
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from poppy.zernike import zernike_basis

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "Masters_Thesis", "figures", "zernike_basis.png")

# Noll index of the mode in each panel, left to right, as in the circular-Bessel
# figure: n=1 cos, sin; n=2 sin2, radial, cos2; n=3 cos3, sin1, cos1, sin3;
# n=4 sin4, cos2, radial, sin2, cos4.
LAYOUT = [[1], [2, 3], [5, 4, 6], [10, 7, 8, 9], [15, 12, 11, 13, 14]]

W_PX, H_PX, PANEL, PITCH, TOP, CENTER = 960, 720, 138, 144, 3, 411

if __name__ == "__main__":
    Z = np.array(zernike_basis(nterms=15, npix=256, outside=0.0))
    fig = plt.figure(figsize=(W_PX / 100, H_PX / 100), dpi=300, facecolor="w")
    for n, row in enumerate(LAYOUT):
        for i, j in enumerate(row):
            x0 = CENTER - 72 * n + PITCH * i
            y0 = TOP + PITCH * n
            ax = fig.add_axes([x0 / W_PX, 1 - (y0 + PANEL) / H_PX, PANEL / W_PX, PANEL / H_PX])
            z = Z[j - 1]
            lim = np.abs(z).max()
            ax.imshow(z, cmap="viridis", vmin=-lim, vmax=lim, interpolation="antialiased")
            ax.set_axis_off()
    fig.savefig(OUT, dpi=300, facecolor="w")
    print(f"wrote {OUT}")
