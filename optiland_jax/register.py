"""Register the JAX backend with Optiland's backend module."""

from __future__ import annotations

import importlib.util
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from optiland.backend.base import AbstractBackend


def jax_to_numpy(obj):
    import jax

    if isinstance(obj, jax.Array):
        return np.asarray(obj)
    raise TypeError


def _patch_ray_tracer_for_jax() -> None:
    """Replace stacked intensity writes with per-surface updates (JAX immutability)."""
    from optiland.raytrace.real_ray_tracer import RealRayTracer
    import optiland.backend as be
    from optiland.rays import PolarizedRays

    if getattr(RealRayTracer, "_jax_patched", False):
        return

    def _write_image_intensity(tracer, rays) -> None:
        tracer.optic.surfaces.surfaces[-1].intensity = be.copy(be.atleast_1d(rays.i))

    def trace_generic(self, Hx, Hy, Px, Py, wavelength):
        self._validate_normalized_coordinates(Hx, Hy, "field")
        self._validate_normalized_coordinates(Px, Py, "pupil")
        vx, vy = self.optic.fields.get_vig_factor(Hx, Hy)
        Px = Px * (1 - vx)
        Py = Py * (1 - vy)
        Hx, Hy, Px, Py = self._validate_array_size(Hx, Hy, Px, Py)
        rays = self.ray_generator.generate_rays(Hx, Hy, Px, Py, wavelength)
        self.optic.surfaces.trace(rays)
        last_surface = self.optic.surfaces[-1]
        last_surface.material_post.propagation_model.propagate(rays, last_surface.thickness)
        _write_image_intensity(self, rays)
        return rays

    def trace(
        self,
        Hx,
        Hy,
        wavelength,
        num_rays: int | None = 100,
        distribution=None,
    ):
        from optiland.distribution import create_distribution

        self._validate_normalized_coordinates(Hx, Hy, "field")
        if isinstance(distribution, str):
            distribution = create_distribution(distribution)
            distribution.generate_points(num_rays)
        Px = distribution.x
        Py = distribution.y
        Hx = be.atleast_1d(Hx)
        Hy = be.atleast_1d(Hy)
        num_fields = len(Hx)
        num_pupil_points = len(Px)
        Hx_full = be.repeat(Hx, num_pupil_points)
        Hy_full = be.repeat(Hy, num_pupil_points)
        Px_full = be.tile(Px, num_fields)
        Py_full = be.tile(Py, num_fields)
        rays = self.ray_generator.generate_rays(Hx_full, Hy_full, Px_full, Py_full, wavelength)
        self.optic.surfaces.trace(rays)
        if self.optic.image_surface:
            last_surface = self.optic.surfaces[-1]
            last_surface.material_post.propagation_model.propagate(rays, last_surface.thickness)
        if isinstance(rays, PolarizedRays):
            rays.update_intensity(self.optic.polarization_state)
        _write_image_intensity(self, rays)
        return rays

    RealRayTracer.trace_generic = trace_generic
    RealRayTracer.trace = trace
    RealRayTracer._jax_patched = True


def _patch_mmdft_for_jax() -> None:
    """Use functional updates when building the MMDFT pupil (JAX immutability)."""
    from optiland.psf.mmdft import MMDFTPSF
    import optiland.backend as be

    if getattr(MMDFTPSF, "_jax_patched", False):
        return

    def _generate_pupil(self):
        x = be.linspace(-1, 1, self.num_rays)
        x, y = be.meshgrid(x, x)
        x = x.ravel()
        y = y.ravel()
        r2 = x**2 + y**2

        field = self.fields[0]
        wl = self.wavelengths[0]
        wavefront_data = self.get_data(field, wl)
        if be.get_backend() == "jax":
            import jax.numpy as jnp

            p = jnp.zeros(x.shape, dtype=jnp.complex128)
        else:
            p = be.to_complex(be.zeros_like(x))

        amplitude = be.sqrt(wavefront_data.intensity)
        phase = be.exp(-1j * 2 * be.pi * wavefront_data.opd)
        values = be.to_complex(amplitude) * phase
        mask = r2 <= 1
        if be.get_backend() == "jax":
            p = p.at[mask].set(values.astype(jnp.complex128))
        else:
            p[mask] = values
        return be.reshape(p, (self.num_rays, self.num_rays))

    MMDFTPSF._generate_pupil = _generate_pupil
    MMDFTPSF._jax_patched = True


def register_jax_backend(force: bool = False) -> None:
    """Inject the JAX backend into ``optiland.backend``.

    Call once at startup::

        import optiland_jax
        optiland_jax.register()
        import optiland.backend as be
        be.set_backend("jax")

    Args:
        force: Replace an existing ``'jax'`` registration if True.
    """
    import optiland.backend as be_module
    from optiland.backend import utils as be_utils
    from optiland_jax.jax_backend import JaxBackend

    if "jax" in be_module._backends and not force:
        return

    be_module._backends["jax"] = JaxBackend()

    import jax

    ndarray_types: list[type] = [np.ndarray, jax.Array]
    if importlib.util.find_spec("torch"):
        import torch

        ndarray_types.insert(1, torch.Tensor)
    be_module.ndarray = tuple(ndarray_types)

    if jax_to_numpy not in be_utils.CONVERTERS:
        be_utils.CONVERTERS.insert(0, jax_to_numpy)

    _orig_array_equal = be_module.array_equal

    def array_equal(a, b) -> bool:
        import jax

        if isinstance(a, jax.Array) or isinstance(b, jax.Array):
            return bool(np.array_equal(np.asarray(a), np.asarray(b)))
        return _orig_array_equal(a, b)

    be_module.array_equal = array_equal

    _patch_ray_tracer_for_jax()
    _patch_mmdft_for_jax()


def list_available_backends() -> list[str]:
    import optiland.backend as be_module

    register_jax_backend()
    return be_module.list_available_backends()
