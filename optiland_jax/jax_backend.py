"""
JAX backend for Optiland — implements AbstractBackend using JAX / jax.numpy.

Kramer Harrison (Optiland), extension for observatory_psf_alignment, 2025
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Any, Literal

import jax
import jax.numpy as jnp
import numpy as np

from optiland.backend.base import AbstractBackend

if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Sequence

    from jax import Array as JaxArray


# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------


class GradMode:
    """Track whether callers expect a differentiable workflow (JAX is always AD-capable)."""

    def __init__(self) -> None:
        self.requires_grad: bool = False

    def enable(self) -> None:
        self.requires_grad = True

    def disable(self) -> None:
        self.requires_grad = False

    @contextlib.contextmanager
    def temporary_enable(self) -> Generator[None, None, None]:
        old = self.requires_grad
        self.requires_grad = True
        try:
            yield
        finally:
            self.requires_grad = old


class _AutogradNamespace:
    """Thin wrapper exposing common JAX AD entry points on ``be.autograd``."""

    grad = staticmethod(jax.grad)
    value_and_grad = staticmethod(jax.value_and_grad)
    jacobian = staticmethod(jax.jacobian)
    hessian = staticmethod(jax.hessian)
    vmap = staticmethod(jax.vmap)


class _Config:
    def __init__(self) -> None:
        self.precision: Literal["float32", "float64"] = "float64"
        self.grad_mode = GradMode()
        self._device = jax.devices("cpu")[0]

    def set_precision(self, precision: Literal["float32", "float64"]) -> None:
        if precision not in ("float32", "float64"):
            raise ValueError("Precision must be 'float32' or 'float64'.")
        self.precision = precision
        jax.config.update("jax_enable_x64", precision == "float64")

    def get_precision(self) -> Literal["float32", "float64"]:
        return self.precision

    def set_device(self, device: str) -> None:
        if device == "cpu":
            devs = jax.devices("cpu")
            if not devs:
                raise ValueError("No JAX CPU device available.")
            self._device = devs[0]
            return
        if device in ("cuda", "gpu"):
            gpu_devs = [d for d in jax.devices() if d.platform != "cpu"]
            if not gpu_devs:
                raise ValueError("No JAX GPU device available.")
            self._device = gpu_devs[0]
            return
        raise ValueError("Device must be 'cpu', 'cuda', or 'gpu'.")

    def get_device(self) -> str:
        return "cpu" if self._device.platform == "cpu" else "cuda"


class JaxBackend(AbstractBackend):
    """Backend implementation using JAX."""

    _lib = jnp

    def __init__(self) -> None:
        self._config = _Config()
        jax.config.update("jax_enable_x64", True)

    # ------------------------------------------------------------------
    # Identity / capabilities
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "jax"

    @property
    def supports_gradients(self) -> bool:
        return True

    @property
    def supports_gpu(self) -> bool:
        return any(d.platform != "cpu" for d in jax.devices())

    @property
    def grad_mode(self) -> GradMode:
        return self._config.grad_mode

    @property
    def autograd(self) -> _AutogradNamespace:
        return _AutogradNamespace()

    @property
    def linalg(self) -> Any:
        return jnp.linalg

    @property
    def fft(self) -> Any:
        return jnp.fft

    @property
    def random(self) -> Any:
        return jax.random

    def set_device(self, device: str) -> None:
        self._config.set_device(device)

    def get_device(self) -> str:
        return self._config.get_device()

    def get_complex_precision(self) -> Any:
        return jnp.complex128 if self._config.precision == "float64" else jnp.complex64

    def to_tensor(self, data: Any, device: str | None = None) -> JaxArray:
        return self.array(data)

    def copy_to(self, source: JaxArray, destination: JaxArray) -> None:
        np.copyto(np.asarray(destination), np.asarray(source))

    # ------------------------------------------------------------------
    # Precision helpers
    # ------------------------------------------------------------------

    def set_precision(self, precision: Literal["float32", "float64"]) -> None:
        self._config.set_precision(precision)

    def get_precision(self) -> int:
        return 32 if self._config.precision == "float32" else 64

    def _dtype(self) -> Any:
        return jnp.float32 if self._config.precision == "float32" else jnp.float64

    def _device(self) -> Any:
        return self._config._device

    def _put(self, x: JaxArray) -> JaxArray:
        return jax.device_put(x, self._device())

    def _resolve_dtype(self, dtype: Any) -> Any:
        if dtype is None:
            return self._dtype()
        if dtype in (np.float32, jnp.float32, "float32"):
            return jnp.float32
        if dtype in (np.float64, jnp.float64, "float64"):
            return jnp.float64
        return dtype

    # ------------------------------------------------------------------
    # Passthrough overrides (shape / IO helpers)
    # ------------------------------------------------------------------

    def shape(self, tensor: JaxArray) -> tuple[int, ...]:
        return tensor.shape

    def size(self, x: JaxArray) -> int:
        return int(x.size)

    def copy(self, x: JaxArray) -> JaxArray:
        return jnp.array(x, copy=True)

    def isscalar(self, x: Any) -> bool:
        return isinstance(x, (int, float)) or (
            isinstance(x, (jax.Array, np.ndarray)) and x.ndim == 0
        )

    def load(self, filename: str) -> JaxArray:
        return self.array(np.load(filename))

    def finfo(self, dtype: Any) -> Any:
        return jnp.finfo(dtype)

    # ------------------------------------------------------------------
    # Array creation
    # ------------------------------------------------------------------

    def array(self, x: Any) -> JaxArray:
        if isinstance(x, jax.Array):
            if jnp.iscomplexobj(x):
                target = self.get_complex_precision()
                arr = x.astype(target) if x.dtype != target else x
                return self._put(arr)
            arr = x.astype(self._dtype()) if x.dtype != self._dtype() else x
            return self._put(arr)
        if isinstance(x, (list, tuple)) and x and any(isinstance(v, jax.Array) for v in x):
            tensors = [self.array(v) for v in x]
            if len({t.ndim for t in tensors}) > 1:
                tensors = [t.reshape(1) if t.ndim == 0 else t for t in tensors]
            try:
                return jnp.stack(tensors)
            except ValueError:
                return jnp.concatenate(tensors)
        return self._put(jnp.asarray(x, dtype=self._dtype()))

    def zeros(self, shape: Sequence[int], dtype: Any = None) -> JaxArray:
        return self._put(jnp.zeros(shape, dtype=self._resolve_dtype(dtype)))

    def ones(self, shape: Sequence[int], dtype: Any = None) -> JaxArray:
        return self._put(jnp.ones(shape, dtype=self._resolve_dtype(dtype)))

    def full(self, shape: Sequence[int], fill_value: Any, dtype: Any = None) -> JaxArray:
        val = self._fill_scalar(fill_value)
        return self._put(jnp.full(shape, val, dtype=self._resolve_dtype(dtype)))

    def linspace(self, start: float, stop: float, num: int = 50) -> JaxArray:
        return self._put(jnp.linspace(start, stop, num, dtype=self._dtype()))

    def arange(self, *args: Any, **kwargs: Any) -> JaxArray:
        return self._put(jnp.arange(*args, dtype=kwargs.pop("dtype", self._dtype()), **kwargs))

    def zeros_like(self, x: Any) -> JaxArray:
        return self._put(jnp.zeros_like(self.array(x), dtype=self._dtype()))

    def ones_like(self, x: Any) -> JaxArray:
        return self._put(jnp.ones_like(self.array(x), dtype=self._dtype()))

    def _fill_scalar(self, fill_value: Any) -> Any:
        if isinstance(fill_value, jax.Array):
            if fill_value.ndim == 0:
                return float(fill_value)
            if fill_value.size == 1:
                return float(fill_value.reshape(-1)[0])
        if isinstance(fill_value, np.ndarray) and fill_value.size == 1:
            return float(fill_value.reshape(-1)[0])
        return fill_value

    def full_like(self, x: Any, fill_value: Any) -> JaxArray:
        x_t = self.array(x)
        fill = self._fill_scalar(fill_value)
        return self._put(jnp.full_like(x_t, fill, dtype=self._dtype()))

    def empty(self, shape: Sequence[int]) -> JaxArray:
        return self._put(jnp.empty(shape, dtype=self._dtype()))

    def empty_like(self, x: Any) -> JaxArray:
        return self._put(jnp.empty_like(self.array(x), dtype=self._dtype()))

    def eye(self, n: int) -> JaxArray:
        return self._put(jnp.eye(n, dtype=self._dtype()))

    def asarray(self, x: Any, **kwargs: Any) -> JaxArray:
        dtype = kwargs.pop("dtype", self._dtype())
        return self._put(jnp.asarray(x, dtype=self._resolve_dtype(dtype), **kwargs))

    # ------------------------------------------------------------------
    # Array utilities
    # ------------------------------------------------------------------

    def cast(self, x: Any) -> JaxArray:
        return self.array(x)

    def is_array_like(self, x: Any) -> bool:
        return isinstance(x, (jax.Array, np.ndarray, list, tuple))

    def arange_indices(self, start: Any, stop: Any = None, step: int = 1) -> JaxArray:
        if stop is None:
            stop = start
            start = 0
        return self._put(jnp.arange(start, stop, step, dtype=jnp.int64))

    def ravel(self, x: Any) -> JaxArray:
        return self.array(x).reshape(-1)

    # ------------------------------------------------------------------
    # Shape / indexing
    # ------------------------------------------------------------------

    def transpose(self, x: Any, axes: Sequence[int] | None = None) -> JaxArray:
        x = self.array(x)
        return jnp.transpose(x, axes) if axes is not None else jnp.transpose(x)

    def reshape(self, x: Any, shape: Sequence[int]) -> JaxArray:
        return self.array(x).reshape(shape)

    def atleast_1d(self, x: Any) -> JaxArray:
        t = self.array(x)
        return t.reshape(1) if t.ndim == 0 else t

    def atleast_2d(self, x: Any) -> JaxArray:
        t = self.array(x)
        if t.ndim == 0:
            return t.reshape(1, 1)
        if t.ndim == 1:
            return t.reshape(1, -1)
        return t

    def as_array_1d(self, data: Any) -> JaxArray:
        if isinstance(data, (int, float)):
            return self.array([data])
        if isinstance(data, (list, tuple)):
            return self.array(data)
        if self.is_array_like(data):
            return self.array(data).reshape(-1)
        raise ValueError("Unsupported type for as_array_1d")

    def stack(self, xs: Sequence[Any], axis: int = 0) -> JaxArray:
        return jnp.stack([self.cast(x) for x in xs], axis=axis)

    def concatenate(self, arrays: Sequence[Any], axis: int = 0) -> JaxArray:
        return jnp.concatenate([self.array(a) for a in arrays], axis=axis)

    def flip(self, x: Any) -> JaxArray:
        return jnp.flip(self.array(x), axis=0)

    def roll(self, x: Any, shift: Any, axis: Any = ()) -> JaxArray:
        ax = None if axis == () else axis
        return jnp.roll(self.array(x), shift, axis=ax)

    def repeat(self, x: Any, repeats: int) -> JaxArray:
        return jnp.repeat(self.array(x), repeats)

    def broadcast_to(self, x: Any, shape: Sequence[int]) -> JaxArray:
        return jnp.broadcast_to(self.array(x), shape)

    def tile(self, x: Any, dims: Any) -> JaxArray:
        d = dims if isinstance(dims, (list, tuple)) else (dims,)
        return jnp.tile(self.array(x), d)

    def expand_dims(self, x: Any, axis: int) -> JaxArray:
        return jnp.expand_dims(self.array(x), axis)

    def meshgrid(self, *arrays: Any) -> tuple[JaxArray, ...]:
        arrs = [self.array(a) for a in arrays]
        return jnp.meshgrid(*arrs, indexing="xy")

    def unsqueeze_last(self, x: Any) -> JaxArray:
        return jnp.expand_dims(self.array(x), -1)

    # ------------------------------------------------------------------
    # Reductions
    # ------------------------------------------------------------------

    def sum(self, x: Any, axis: int | None = None) -> JaxArray:
        x = self.array(x)
        return jnp.sum(x, axis=axis) if axis is not None else jnp.sum(x)

    def mean(self, x: Any, axis: int | None = None, keepdims: bool = False) -> JaxArray:
        return jnp.nanmean(self.array(x), axis=axis, keepdims=keepdims)

    def std(self, x: Any, axis: int | None = None) -> JaxArray:
        x = self.array(x)
        return jnp.std(x, axis=axis) if axis is not None else jnp.std(x)

    def max(self, x: Any) -> float:
        return float(jnp.max(self.array(x)))

    def min(self, x: Any) -> float:
        return float(jnp.min(self.array(x)))

    def argmin(self, x: Any, axis: int | None = None) -> JaxArray:
        return jnp.argmin(self.array(x), axis=axis)

    def argwhere(self, x: Any) -> JaxArray:
        return jnp.argwhere(self.array(x))

    def clip(self, x: Any, a_min: Any, a_max: Any) -> JaxArray:
        return jnp.clip(self.array(x), a_min, a_max)

    def where(self, condition: Any, x: Any, y: Any) -> JaxArray:
        if isinstance(condition, bool):
            return self.array(x if condition else y)
        return jnp.where(condition, self.array(x), self.array(y))

    def maximum(self, a: Any, b: Any) -> JaxArray:
        return jnp.maximum(self.array(a), self.array(b))

    def minimum(self, a: Any, b: Any) -> JaxArray:
        return jnp.minimum(self.array(a), self.array(b))

    def fmax(self, a: Any, b: Any) -> JaxArray:
        return jnp.fmax(self.array(a), self.array(b))

    def power(self, x: Any, y: Any) -> JaxArray:
        return jnp.power(self.array(x), self.array(y))

    def diff(self, x: Any, n: int = 1, axis: int = -1, **kwargs: Any) -> JaxArray:
        return jnp.diff(self.array(x), n=n, axis=axis, **kwargs)

    def all(self, x: Any) -> bool:
        if isinstance(x, bool):
            return x
        return bool(jnp.all(self.array(x)))

    def any(self, x: Any) -> bool:
        if isinstance(x, bool):
            return x
        return bool(jnp.any(self.array(x)))

    def nanmax(self, x: Any, axis: int | None = None, keepdim: bool = False) -> JaxArray:
        return jnp.nanmax(self.array(x), axis=axis, keepdims=keepdim)

    def sort(self, x: Any, axis: int = -1) -> JaxArray:
        return jnp.sort(self.array(x), axis=axis)

    def isclose(self, a: Any, b: Any, rtol: float = 1e-5, atol: float = 1e-8) -> JaxArray:
        return jnp.isclose(self.array(a), self.array(b), rtol=rtol, atol=atol)

    def allclose(self, a: Any, b: Any, rtol: float = 1e-5, atol: float = 1e-8) -> bool:
        return bool(jnp.allclose(self.array(a), self.array(b), rtol=rtol, atol=atol))

    def radians(self, x: Any) -> JaxArray:
        return jnp.deg2rad(self.array(x))

    def degrees(self, x: Any) -> JaxArray:
        return jnp.rad2deg(self.array(x))

    # ------------------------------------------------------------------
    # Elementwise passthrough wrappers (accept Python scalars / lists)
    # ------------------------------------------------------------------

    def _u(self, x: Any) -> JaxArray:
        return self.array(x)

    def sin(self, x: Any) -> JaxArray:
        return jnp.sin(self._u(x))

    def cos(self, x: Any) -> JaxArray:
        return jnp.cos(self._u(x))

    def tan(self, x: Any) -> JaxArray:
        return jnp.tan(self._u(x))

    def arcsin(self, x: Any) -> JaxArray:
        return jnp.arcsin(self._u(x))

    def arccos(self, x: Any) -> JaxArray:
        return jnp.arccos(self._u(x))

    def arctan(self, x: Any) -> JaxArray:
        return jnp.arctan(self._u(x))

    def arctan2(self, y: Any, x: Any) -> JaxArray:
        return jnp.arctan2(self._u(y), self._u(x))

    def sinh(self, x: Any) -> JaxArray:
        return jnp.sinh(self._u(x))

    def cosh(self, x: Any) -> JaxArray:
        return jnp.cosh(self._u(x))

    def tanh(self, x: Any) -> JaxArray:
        return jnp.tanh(self._u(x))

    def exp(self, x: Any) -> JaxArray:
        x = self._u(x)
        if jnp.iscomplexobj(x):
            return jnp.exp(x)
        return jnp.exp(x)

    def log(self, x: Any) -> JaxArray:
        return jnp.log(self._u(x))

    def log2(self, x: Any) -> JaxArray:
        return jnp.log2(self._u(x))

    def log10(self, x: Any) -> JaxArray:
        return jnp.log10(self._u(x))

    def sqrt(self, x: Any) -> JaxArray:
        return jnp.sqrt(self._u(x))

    def abs(self, x: Any) -> JaxArray:
        return jnp.abs(self._u(x))

    def sign(self, x: Any) -> JaxArray:
        return jnp.sign(self._u(x))

    def floor(self, x: Any) -> JaxArray:
        return jnp.floor(self._u(x))

    def ceil(self, x: Any) -> JaxArray:
        return jnp.ceil(self._u(x))

    def hypot(self, x: Any, y: Any) -> JaxArray:
        return jnp.hypot(self._u(x), self._u(y))

    def deg2rad(self, x: Any) -> JaxArray:
        return jnp.deg2rad(self._u(x))

    def rad2deg(self, x: Any) -> JaxArray:
        return jnp.rad2deg(self._u(x))

    def conj(self, x: Any) -> JaxArray:
        return jnp.conj(self._u(x))

    def real(self, x: Any) -> JaxArray:
        return jnp.real(self._u(x))

    def imag(self, x: Any) -> JaxArray:
        return jnp.imag(self._u(x))

    def copysign(self, x: Any, y: Any) -> JaxArray:
        return jnp.copysign(self._u(x), self._u(y))

    def isnan(self, x: Any) -> Any:
        return jnp.isnan(self._u(x))

    def isinf(self, x: Any) -> Any:
        return jnp.isinf(self._u(x))

    def isfinite(self, x: Any) -> Any:
        return jnp.isfinite(self._u(x))

    # ------------------------------------------------------------------
    # Linear algebra
    # ------------------------------------------------------------------

    def matmul(self, a: Any, b: Any) -> JaxArray:
        a, b = self.array(a), self.array(b)
        dtype = jnp.result_type(a, b)
        return jnp.matmul(a.astype(dtype), b.astype(dtype))

    def cross(
        self,
        a: Any,
        b: Any,
        axisa: int = -1,
        axisb: int = -1,
        axisc: int = -1,
        axis: int | None = None,
    ) -> JaxArray:
        if axis is not None:
            axisa = axisb = axisc = axis
        return jnp.cross(self.array(a), self.array(b), axisa=axisa, axisb=axisb, axisc=axisc)

    def batched_chain_matmul3(self, a: Any, b: Any, c: Any) -> JaxArray:
        a, b, c = self.array(a), self.array(b), self.array(c)
        dtype = jnp.result_type(a, b, c)
        a, b, c = a.astype(dtype), b.astype(dtype), c.astype(dtype)
        return jnp.matmul(jnp.matmul(a, b), c)

    def matrix_vector_multiply_and_squeeze(self, p: Any, E: Any) -> JaxArray:
        p, e = self.array(p), self.array(E)
        return jnp.squeeze(jnp.matmul(p, e[:, :, jnp.newaxis]), axis=2)

    def mult_p_E(self, p: Any, E: Any) -> JaxArray:
        p_c = self.array(p).astype(jnp.complex128)
        e_c = self.array(E).astype(jnp.complex128)
        return jnp.squeeze(jnp.matmul(p_c, e_c[:, :, jnp.newaxis]), axis=2)

    def lstsq(self, a: Any, b: Any) -> JaxArray:
        return jnp.linalg.lstsq(self.array(a), self.array(b))[0]

    def to_complex(self, x: Any) -> JaxArray:
        return self.array(x).astype(jnp.complex128)

    def dot(self, a: Any, b: Any) -> JaxArray:
        return jnp.dot(self.array(a), self.array(b))

    def outer(self, a: Any, b: Any) -> JaxArray:
        return jnp.outer(self.array(a), self.array(b))

    def einsum(self, subscripts: str, *operands: Any, **kwargs: Any) -> JaxArray:
        ops = [self.array(o) for o in operands]
        return jnp.einsum(subscripts, *ops, **kwargs)

    # ------------------------------------------------------------------
    # Interpolation
    # ------------------------------------------------------------------

    def interp(self, x: Any, xp: Any, fp: Any) -> JaxArray:
        x = self.array(x)
        xp = self.array(xp)
        fp = self.array(fp)
        order = jnp.argsort(xp)
        xp = xp[order]
        fp = fp[order]
        x_clip = jnp.clip(x, xp[0], xp[-1])
        idx = jnp.searchsorted(xp, x_clip, side="right")
        idx = jnp.clip(idx, 1, xp.shape[0] - 1)
        x0, x1 = xp[idx - 1], xp[idx]
        y0, y1 = fp[idx - 1], fp[idx]
        return y0 + (y1 - y0) * (x_clip - x0) / (x1 - x0)

    def nearest_nd_interpolator(
        self, points: Any, values: Any, Hx: Any, Hy: Any
    ) -> JaxArray:
        if Hx is None or Hy is None:
            raise ValueError("Hx and Hy must be provided")
        points = self.array(points)
        values = self.array(values)
        hx, hy = self.array(Hx), self.array(Hy)
        hx, hy = jnp.broadcast_arrays(hx, hy)
        q_flat = jnp.stack([hx, hy], axis=-1).reshape(-1, 2)
        d = jnp.linalg.norm(q_flat[:, None, :] - points[None, :, :], axis=-1)
        idx = jnp.argmin(d, axis=1)
        vals = values.reshape(points.shape[0], -1)
        out = vals[idx].reshape(*hx.shape, -1)
        return out[..., 0] if out.shape[-1] == 1 else out

    def grid_sample(
        self,
        input: Any,
        grid: Any,
        mode: str = "bilinear",
        padding_mode: str = "zeros",
        align_corners: bool = False,
    ) -> JaxArray:
        inp = self.array(input)
        grd = self.array(grid)
        n, c, h_in, w_in = inp.shape
        h_out, w_out = grd.shape[1], grd.shape[2]
        x = grd[..., 0]
        y = grd[..., 1]

        if align_corners:
            x_pix = ((x + 1) / 2) * (w_in - 1)
            y_pix = ((y + 1) / 2) * (h_in - 1)
        else:
            x_pix = ((x + 1) * w_in / 2) - 0.5
            y_pix = ((y + 1) * h_in / 2) - 0.5

        def _sample_plane(plane: JaxArray, xp: JaxArray, yp: JaxArray) -> JaxArray:
            if mode == "nearest":
                xi = jnp.clip(jnp.round(xp).astype(jnp.int32), 0, w_in - 1)
                yi = jnp.clip(jnp.round(yp).astype(jnp.int32), 0, h_in - 1)
                return plane[yi, xi]
            x0 = jnp.floor(xp).astype(jnp.int32)
            y0 = jnp.floor(yp).astype(jnp.int32)
            x1 = jnp.clip(x0 + 1, 0, w_in - 1)
            y1 = jnp.clip(y0 + 1, 0, h_in - 1)
            x0 = jnp.clip(x0, 0, w_in - 1)
            y0 = jnp.clip(y0, 0, h_in - 1)
            wx = xp - x0
            wy = yp - y0
            v00 = plane[y0, x0]
            v01 = plane[y1, x0]
            v10 = plane[y0, x1]
            v11 = plane[y1, x1]
            return (1 - wx) * (1 - wy) * v00 + (1 - wx) * wy * v01 + wx * (1 - wy) * v10 + wx * wy * v11

        outputs = []
        for batch in range(n):
            batch_out = []
            for ch in range(c):
                plane = inp[batch, ch]
                xp, yp = x_pix[batch], y_pix[batch]
                if padding_mode == "zeros":

                    def _pad_sample(xp_i, yp_i):
                        inside = (xp_i >= 0) & (xp_i <= w_in - 1) & (yp_i >= 0) & (yp_i <= h_in - 1)
                        val = _sample_plane(plane, xp_i, yp_i)
                        return jnp.where(inside, val, 0.0)

                    sampled = jax.vmap(_pad_sample)(xp.ravel(), yp.ravel()).reshape(h_out, w_out)
                else:
                    xp_c = jnp.clip(xp, 0, w_in - 1)
                    yp_c = jnp.clip(yp, 0, h_in - 1)
                    sampled = _sample_plane(plane, xp_c, yp_c)
                batch_out.append(sampled)
            outputs.append(jnp.stack(batch_out, axis=0))
        return jnp.stack(outputs, axis=0)

    def get_bilinear_weights(
        self, coords: Any, bin_edges: Sequence[Any]
    ) -> tuple[JaxArray, JaxArray]:
        x_edges, y_edges = bin_edges[0], bin_edges[1]
        x_edges, y_edges = self.array(x_edges), self.array(y_edges)
        coords = self.array(coords)
        x = coords[:, 0]
        y = coords[:, 1]

        valid = (x >= x_edges[0]) & (x <= x_edges[-1]) & (y >= y_edges[0]) & (y <= y_edges[-1])
        x_centers = (x_edges[:-1] + x_edges[1:]) / 2
        y_centers = (y_edges[:-1] + y_edges[1:]) / 2

        ix = jnp.searchsorted(x_centers, x, side="right") - 1
        iy = jnp.searchsorted(y_centers, y, side="right") - 1
        ix = jnp.clip(ix, 0, x_centers.shape[0] - 2)
        iy = jnp.clip(iy, 0, y_centers.shape[0] - 2)

        x0, x1 = x_centers[ix], x_centers[ix + 1]
        y0, y1 = y_centers[iy], y_centers[iy + 1]
        wx = (x - x0) / (x1 - x0 + 1e-9)
        wy = (y - y0) / (y1 - y0 + 1e-9)

        w00 = (1 - wx) * (1 - wy)
        w01 = (1 - wx) * wy
        w10 = wx * (1 - wy)
        w11 = wx * wy

        all_indices = jnp.stack(
            [
                jnp.stack([ix, iy], axis=1),
                jnp.stack([ix, iy + 1], axis=1),
                jnp.stack([ix + 1, iy], axis=1),
                jnp.stack([ix + 1, iy + 1], axis=1),
            ],
            axis=1,
        )
        all_weights = jnp.stack([w00, w01, w10, w11], axis=1)
        all_weights = all_weights * valid[:, None].astype(all_weights.dtype)
        return all_indices, all_weights

    # ------------------------------------------------------------------
    # Polynomial / signal
    # ------------------------------------------------------------------

    def polyfit(self, x: Any, y: Any, degree: int) -> JaxArray:
        x = self.array(x)
        y = self.array(y)
        x_mat = jnp.stack([x**i for i in range(degree, -1, -1)], axis=1)
        return jnp.linalg.lstsq(x_mat, y[:, None] if y.ndim == 1 else y)[0][: degree + 1].squeeze()

    def polyval(self, coeffs: Any, x: Any) -> JaxArray:
        coeffs = self.array(coeffs)
        x = self.array(x)
        return sum(c * x**i for i, c in enumerate(reversed(list(coeffs))))

    def fftconvolve(
        self, in1: Any, in2: Any, mode: Literal["full", "valid", "same"] = "full"
    ) -> JaxArray:
        a = self.array(in1)
        b = self.array(in2)
        if a.ndim != b.ndim:
            raise ValueError("Inputs must have the same dimensionality.")
        s1, s2 = a.shape, b.shape
        shape = tuple(s1[i] + s2[i] - 1 for i in range(a.ndim))
        fa = jnp.fft.fftn(a, s=shape)
        fb = jnp.fft.fftn(b, s=shape)
        ret = jnp.real(jnp.fft.ifftn(fa * fb, s=shape))
        if mode == "full":
            return ret
        if mode == "same":
            slices = tuple(slice((s2[i] - 1) // 2, (s2[i] - 1) // 2 + s1[i]) for i in range(a.ndim))
            return ret[slices]
        if mode == "valid":
            slices = tuple(slice(s2[i] - 1, s1[i]) for i in range(a.ndim))
            return ret[slices]
        raise ValueError(f"Unknown mode: {mode}")

    # ------------------------------------------------------------------
    # Random
    # ------------------------------------------------------------------

    def default_rng(self, seed: int | None = None) -> Any:
        return jax.random.PRNGKey(0 if seed is None else seed)

    def random_uniform(
        self,
        low: float = 0.0,
        high: float = 1.0,
        size: Any = None,
        generator: Any = None,
    ) -> JaxArray:
        key = generator if generator is not None else jax.random.PRNGKey(0)
        key, sub = jax.random.split(key)
        return low + (high - low) * jax.random.uniform(sub, shape=size or (), dtype=self._dtype())

    def rand(self, *size: int) -> JaxArray:
        shape = size or (1,)
        key = jax.random.PRNGKey(jax.random.randint(jax.random.PRNGKey(0), (), 0, 2**30))
        return jax.random.uniform(key, shape=shape, dtype=self._dtype())

    def random_normal(
        self,
        loc: float = 0.0,
        scale: float = 1.0,
        size: Any = None,
        generator: Any = None,
    ) -> JaxArray:
        key = generator if generator is not None else jax.random.PRNGKey(0)
        key, sub = jax.random.split(key)
        return loc + scale * jax.random.normal(sub, shape=size or (), dtype=self._dtype())

    def sobol_sampler(
        self,
        dim: int,
        num_samples: int,
        scramble: bool = True,
        seed: int | None = None,
    ) -> JaxArray:
        try:
            from scipy.stats import qmc
        except ImportError as exc:
            raise ImportError("scipy is required for Sobol sampling with jax backend") from exc
        if num_samples > 0:
            num_samples_pow2 = 1 << (num_samples - 1).bit_length()
        else:
            num_samples_pow2 = num_samples
        sobol = qmc.Sobol(d=dim, scramble=scramble, seed=seed)
        samples = sobol.random(n=num_samples_pow2)
        return self.array(samples[:num_samples].astype(np.float32))

    def erfinv(self, x: Any) -> JaxArray:
        import jax.scipy.special as jsp

        return jsp.erfinv(self.array(x))

    # ------------------------------------------------------------------
    # Miscellaneous
    # ------------------------------------------------------------------

    def factorial(self, n: Any) -> JaxArray:
        import jax.scipy.special as jsp

        return jsp.gamma(self.array(n) + 1)

    def path_contains_points(self, vertices: Any, points: Any) -> JaxArray:
        vx, vy = self.array(vertices)[:, 0], self.array(vertices)[:, 1]
        px = self.array(points)[:, 0][:, None]
        py = self.array(points)[:, 1][:, None]
        vx_next = jnp.roll(vx, -1)
        vy_next = jnp.roll(vy, -1)
        cond = (vy > py) != (vy_next > py)
        slope = (vx_next - vx) / (vy_next - vy)
        x_int = vx + slope * (py - vy)
        cross = cond & (px < x_int)
        return (jnp.sum(cross, axis=1) % 2) == 1

    def pad(
        self,
        tensor: Any,
        pad_width: Any,
        mode: str = "constant",
        constant_values: float | None = 0,
    ) -> JaxArray:
        if mode != "constant":
            raise NotImplementedError("Only constant mode supported")
        return jnp.pad(
            self.array(tensor),
            pad_width,
            mode="constant",
            constant_values=constant_values,
        )

    def vectorize(self, pyfunc: Callable[..., Any]) -> Callable[..., Any]:
        return jax.vmap(lambda v: pyfunc(v))

    @contextlib.contextmanager
    def errstate(self, **kwargs: Any) -> Generator[None, None, None]:
        yield

    def histogram(self, x: Any, bins: Any = 10) -> tuple[JaxArray, JaxArray]:
        x_np = np.asarray(self.array(x))
        if isinstance(bins, int):
            counts, edges = np.histogram(x_np, bins=bins)
        else:
            counts, edges = np.histogram(x_np, bins=np.asarray(bins))
        return self.array(counts), self.array(edges)

    def histogram2d(
        self,
        x: Any,
        y: Any,
        bins: Any,
        weights: Any | None = None,
    ) -> tuple[JaxArray, JaxArray, JaxArray]:
        if not isinstance(bins, (list, tuple)) or len(bins) != 2:
            raise ValueError("`bins` must be a list or tuple of two edge arrays.")
        x_edges, y_edges = self.array(bins[0]), self.array(bins[1])
        nx = x_edges.shape[0] - 1
        ny = y_edges.shape[0] - 1
        x = self.array(x)
        y = self.array(y)
        x_idx = jnp.searchsorted(x_edges, x, side="left") - 1
        y_idx = jnp.searchsorted(y_edges, y, side="left") - 1
        x_idx = jnp.clip(x_idx, 0, nx - 1)
        y_idx = jnp.clip(y_idx, 0, ny - 1)
        mask = (x >= x_edges[0]) & (x <= x_edges[-1]) & (y >= y_edges[0]) & (y <= y_edges[-1])
        w = self.array(weights) if weights is not None else jnp.ones_like(x)
        valid_x = x_idx[mask]
        valid_y = y_idx[mask]
        valid_w = w[mask]
        linear = (valid_y * nx + valid_x).astype(jnp.int32)
        hist_flat = jnp.zeros(nx * ny, dtype=valid_w.dtype)
        hist_flat = hist_flat.at[linear].add(valid_w)
        hist = hist_flat.reshape(ny, nx).T
        return hist, x_edges, y_edges

    def nansum(self, x: Any, axis: int | None = None) -> JaxArray:
        return jnp.nansum(self.array(x), axis=axis)

    def nanmean(self, x: Any, axis: int | None = None) -> JaxArray:
        return jnp.nanmean(self.array(x), axis=axis)

    def vstack(self, xs: Sequence[Any]) -> JaxArray:
        return jnp.vstack([self.array(x) for x in xs])

    def column_stack(self, xs: Sequence[Any]) -> JaxArray:
        return jnp.column_stack([self.array(x) for x in xs])

    def searchsorted(self, a: Any, v: Any, side: str = "left") -> JaxArray:
        return jnp.searchsorted(self.array(a), self.array(v), side=side)

    def round(self, x: Any, decimals: int = 0) -> JaxArray:
        return jnp.round(self.array(x), decimals=decimals)
