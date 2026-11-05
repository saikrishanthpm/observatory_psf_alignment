"""Register the MLX backend with Optiland's backend module."""

from __future__ import annotations

import importlib.util

import numpy as np

from optiland_mlx import _helpers as h


def mlx_to_numpy(obj):
    if h.is_mx_array(obj):
        return np.asarray(obj)
    raise TypeError


def _patch_mlx_array_compat() -> None:
    """MLX array API gaps used directly by Optiland (not via be.*)."""
    import mlx.core as mx

    if getattr(mx.array, "_optiland_compat_patched", False):
        return

    def _ravel(self):
        return self.reshape(-1)

    def _flatten(self):
        return self.reshape(-1)

    _orig_astype = mx.array.astype

    def _astype(self, dtype, stream=None):
        if dtype is bool:
            dtype = mx.bool_
        elif dtype is int:
            dtype = mx.int64
        elif dtype is float:
            dtype = mx.float64
        if stream is None:
            return _orig_astype(self, dtype)
        return _orig_astype(self, dtype, stream=stream)

    mx.array.ravel = _ravel
    mx.array.flatten = _flatten
    mx.array.astype = _astype

    _orig_getitem = mx.array.__getitem__
    _orig_setitem = mx.array.__setitem__

    def _getitem(self, key):
        if isinstance(key, mx.array) and mx.issubdtype(key.dtype, mx.bool_):
            flat = np.asarray(self).ravel()
            return mx.array(flat[np.asarray(key).ravel()])
        if isinstance(key, np.ndarray) and key.dtype == bool:
            flat = np.asarray(self).ravel()
            return mx.array(flat[key.ravel()])
        return _orig_getitem(self, key)

    def _setitem(self, key, value):
        if isinstance(key, mx.array) and mx.issubdtype(key.dtype, mx.bool_):
            data = np.asarray(self).copy()
            data[np.asarray(key)] = np.asarray(value)
            np.copyto(np.asarray(self), data)
            return
        if isinstance(key, np.ndarray) and key.dtype == bool:
            data = np.asarray(self).copy()
            data[key] = np.asarray(value)
            np.copyto(np.asarray(self), data)
            return
        return _orig_setitem(self, key, value)

    mx.array.__getitem__ = _getitem
    mx.array.__setitem__ = _setitem
    mx.array._optiland_compat_patched = True


def _patch_ray_tracer() -> None:
    from optiland.raytrace.real_ray_tracer import RealRayTracer
    import optiland.backend as be
    from optiland.rays import PolarizedRays

    if getattr(RealRayTracer, "_mlx_patched", False):
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

    def trace(self, Hx, Hy, wavelength, num_rays: int | None = 100, distribution=None):
        from optiland.distribution import create_distribution

        self._validate_normalized_coordinates(Hx, Hy, "field")
        if isinstance(distribution, str):
            distribution = create_distribution(distribution)
            distribution.generate_points(num_rays)
        Px, Py = distribution.x, distribution.y
        Hx, Hy = be.atleast_1d(Hx), be.atleast_1d(Hy)
        n_f, n_p = len(Hx), len(Px)
        rays = self.ray_generator.generate_rays(
            be.repeat(Hx, n_p), be.repeat(Hy, n_p), be.tile(Px, n_f), be.tile(Py, n_f), wavelength
        )
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
    RealRayTracer._mlx_patched = True


def _patch_mmdft() -> None:
    from optiland.psf.mmdft import MMDFTPSF
    import optiland.backend as be

    if getattr(MMDFTPSF, "_mlx_patched", False):
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
        amplitude = be.sqrt(wavefront_data.intensity)
        phase = be.exp(-1j * 2 * be.pi * wavefront_data.opd)
        values = be.to_complex(amplitude) * phase
        mask = r2 <= 1
        if be.get_backend() == "mlx":
            p_flat = np.zeros(int(be.size(x)), dtype=np.complex128)
            p_flat[np.asarray(mask)] = np.asarray(values)
            p = be.array(p_flat)
        else:
            p = be.to_complex(be.zeros_like(x))
            p[mask] = values
        return be.reshape(p, (self.num_rays, self.num_rays))

    MMDFTPSF._generate_pupil = _generate_pupil
    MMDFTPSF._mlx_patched = True


def register_mlx_backend(force: bool = False) -> None:
    """Inject the MLX backend into ``optiland.backend``.

    Call once at startup::

        import optiland_mlx
        optiland_mlx.register()
        import optiland.backend as be
        be.set_backend("mlx")
    """
    import optiland.backend as be_module
    from optiland.backend import utils as be_utils
    from optiland_mlx.mlx_backend import MlxBackend

    if "mlx" in be_module._backends and not force:
        return

    be_module._backends["mlx"] = MlxBackend()

    import mlx.core as mx

    ndarray_types: list[type] = [np.ndarray, mx.array]
    if importlib.util.find_spec("torch"):
        import torch

        ndarray_types.insert(1, torch.Tensor)
    if importlib.util.find_spec("jax"):
        import jax

        ndarray_types.append(jax.Array)
    be_module.ndarray = tuple(ndarray_types)

    if mlx_to_numpy not in be_utils.CONVERTERS:
        be_utils.CONVERTERS.insert(0, mlx_to_numpy)

    _orig_array_equal = be_module.array_equal

    def array_equal(a, b) -> bool:
        import mlx.core as mx

        if isinstance(a, mx.array) or isinstance(b, mx.array):
            return bool(np.array_equal(np.asarray(a), np.asarray(b)))
        return _orig_array_equal(a, b)

    be_module.array_equal = array_equal

    _patch_mlx_array_compat()
    _patch_ray_tracer()
    _patch_mmdft()


def list_available_backends() -> list[str]:
    import optiland.backend as be_module

    register_mlx_backend()
    return be_module.list_available_backends()
