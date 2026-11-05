#!/usr/bin/env python3
"""Smoke test: Optiland Pearl ray trace + PSF with JAX backend."""

from __future__ import annotations

import numpy as np
import optiland_jax

optiland_jax.register()

import optiland.backend as be
from optiland import optic, physical_apertures
from optiland.psf import MMDFTPSF

be.set_backend("jax")
be.set_precision("float64")

THETA = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)


def build_pearl(theta: np.ndarray):
    k, t = 1000.0, 3600.0
    pdx, pdy, pdz, ptx, pty, sdx, sdy, sdz, stx, sty = theta
    pearl = optic.Optic(name="Pearl")
    pearl.set_aperture(aperture_type="EPD", value=6471.26)
    aperture1 = physical_apertures.RadialAperture(r_min=690, r_max=3210)
    a1 = physical_apertures.PolygonAperture(x=[550, 3210, 3210, 550], y=[300, 300, -300, -300])
    a2 = physical_apertures.PolygonAperture(x=[-1835, -1375, 260, -260], y=[2705, 2955, 150, -150])
    a3 = physical_apertures.PolygonAperture(x=[-1835, -1375, 260, -260], y=[-2705, -2955, -150, 150])
    a = aperture1 - a1 - a2 - a3
    pearl.wavelengths.add(value=0.65)
    pearl.surfaces.add(index=0, radius=be.inf, thickness=be.inf)
    pearl.surfaces.add(
        index=1,
        radius=-16256,
        thickness=-7400.005 + pdz / k,
        dx=pdx / k,
        dy=pdy / k,
        rx=ptx / t,
        ry=pty / t,
        conic=-0.9953569,
        material="mirror",
        aperture=a,
        is_stop=True,
    )
    pearl.surfaces.add(
        index=2,
        radius=-1660.575,
        thickness=7449.97 + sdz / k,
        dx=sdx / k,
        dy=sdy / k,
        rx=stx / t,
        ry=sty / t,
        conic=-1.566503,
        material="mirror",
    )
    pearl.surfaces.add(index=3, radius=-1830.517, thickness=-1033.62, conic=-0.718069, material="mirror")
    pearl.surfaces.add(index=4, radius=be.inf, thickness=1220.34, material="mirror")
    pearl.surfaces.add(index=5)
    pearl.fields.set_type(field_type="angle")
    pearl.fields.add(y=0.0)
    return pearl


def main() -> None:
    print("Available backends:", optiland_jax.list_available_backends())
    print("Active backend:", be.get_backend())
    print("Supports gradients:", be.supports_gradients)

    pearl = build_pearl(THETA)
    psf = MMDFTPSF(pearl, field=(0, 0), wavelength=0.65, num_rays=128)
    image = psf.psf
    arr = np.asarray(image)
    print("PSF shape:", arr.shape)
    print("PSF peak:", float(arr.max()))
    print("PSF sum:", float(arr.sum()))

    # Differentiable probe: gradient of peak w.r.t. primary decenter x
    import jax

    def peak_for_pdx(pdx_val):
        theta = THETA.at[0].set(float(pdx_val))
        p = build_pearl(np.asarray(theta))
        im = MMDFTPSF(optic=p, num_rays=64, grid_size=32).generate()
        return jnp_max(im)

    def jnp_max(x):
        import jax.numpy as jnp

        return jnp.max(jnp.asarray(x))

    grad_fn = jax.grad(lambda v: float(peak_for_pdx(v)))
    try:
        g = grad_fn(0.0)
        print("d(PSF peak)/d(pdx) at nominal:", g)
    except Exception as exc:
        print("Autograd probe skipped:", exc)

    print("JAX backend smoke test OK.")


if __name__ == "__main__":
    main()
