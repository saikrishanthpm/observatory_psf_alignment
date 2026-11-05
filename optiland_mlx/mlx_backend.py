"""
MLX backend for Optiland — implements AbstractBackend using Apple MLX.

Extension for observatory_psf_alignment / spaceklip_new kernel.
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Any, Literal

import mlx.core as mx
import numpy as np
from scipy.ndimage import map_coordinates
from scipy.signal import fftconvolve as _scipy_fftconvolve

from optiland.backend.base import AbstractBackend
from optiland_mlx import _helpers as h

if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Sequence


class GradMode:
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
    grad = staticmethod(mx.grad)
    value_and_grad = staticmethod(mx.value_and_grad)


class _Config:
    def __init__(self) -> None:
        self.precision: Literal["float32", "float64"] = "float64"
        self.grad_mode = GradMode()
        self._apply_device_for_precision()

    def set_precision(self, precision: Literal["float32", "float64"]) -> None:
        if precision not in ("float32", "float64"):
            raise ValueError("Precision must be 'float32' or 'float64'.")
        self.precision = precision
        self._apply_device_for_precision()

    def _apply_device_for_precision(self) -> None:
        # MLX float64 is CPU-only; float32 can use the GPU.
        if self.precision == "float64":
            mx.set_default_device(mx.cpu)
        else:
            mx.set_default_device(mx.gpu)

    def get_precision(self) -> Literal["float32", "float64"]:
        return self.precision


class MlxBackend(AbstractBackend):
    """Backend implementation using Apple MLX."""

    _lib = mx

    def __init__(self) -> None:
        self._config = _Config()

    @property
    def name(self) -> str:
        return "mlx"

    @property
    def supports_gradients(self) -> bool:
        return True

    @property
    def supports_gpu(self) -> bool:
        return True

    @property
    def grad_mode(self) -> GradMode:
        return self._config.grad_mode

    @property
    def autograd(self) -> _AutogradNamespace:
        return _AutogradNamespace()

    @property
    def linalg(self) -> Any:
        return mx.linalg

    @property
    def fft(self) -> Any:
        return mx.fft

    @property
    def random(self) -> Any:
        return mx.random

    def set_device(self, device: str) -> None:
        if device in ("cpu", "CPU"):
            mx.set_default_device(mx.cpu)
        elif device in ("gpu", "cuda", "mps", "GPU"):
            mx.set_default_device(mx.gpu)
        else:
            raise ValueError("Device must be 'cpu' or 'gpu'.")

    def get_device(self) -> str:
        return "cpu" if self._config.precision == "float64" else "gpu"

    def get_complex_precision(self) -> Any:
        return mx.complex64

    def to_tensor(self, data: Any, device: str | None = None) -> mx.array:
        return self.array(data)

    def copy_to(self, source: mx.array, destination: mx.array) -> None:
        np.copyto(h.to_numpy(destination), h.to_numpy(source))

    def set_precision(self, precision: Literal["float32", "float64"]) -> None:
        self._config.set_precision(precision)

    def get_precision(self) -> int:
        return 32 if self._config.precision == "float32" else 64

    def _dtype(self) -> Any:
        return mx.float32 if self._config.precision == "float32" else mx.float64

    def _cdtype(self) -> Any:
        return mx.complex64

    def _resolve_dtype(self, dtype: Any) -> Any:
        if dtype is None:
            return self._dtype()
        if dtype in (np.float32, mx.float32, "float32"):
            return mx.float32
        if dtype in (np.float64, mx.float64, "float64"):
            return mx.float64
        return dtype

    def shape(self, tensor: mx.array) -> tuple[int, ...]:
        return tensor.shape

    def size(self, x: mx.array) -> int:
        return int(x.size)

    def copy(self, x: mx.array) -> mx.array:
        return mx.array(x)

    def isscalar(self, x: Any) -> bool:
        return isinstance(x, (int, float)) or (isinstance(x, mx.array) and x.ndim == 0)

    def load(self, filename: str) -> mx.array:
        return self.array(np.load(filename))

    def finfo(self, dtype: Any) -> Any:
        return np.finfo(h.to_numpy(self.array([0.0])).dtype)

    def array(self, x: Any) -> mx.array:
        if isinstance(x, mx.array):
            if mx.issubdtype(x.dtype, mx.complexfloating):
                return x.astype(self._cdtype()) if x.dtype != self._cdtype() else x
            return x.astype(self._dtype()) if x.dtype != self._dtype() else x
        if isinstance(x, (list, tuple)) and x and any(isinstance(v, mx.array) for v in x):
            tensors = [self.array(v) for v in x]
            if len({t.ndim for t in tensors}) > 1:
                tensors = [t.reshape(1) if t.ndim == 0 else t for t in tensors]
            try:
                return mx.stack(tensors)
            except ValueError:
                return mx.concatenate(tensors)
        return mx.array(x, dtype=self._dtype())

    def zeros(self, shape: Sequence[int], dtype: Any = None) -> mx.array:
        return mx.zeros(shape, dtype=self._resolve_dtype(dtype))

    def ones(self, shape: Sequence[int], dtype: Any = None) -> mx.array:
        return mx.ones(shape, dtype=self._resolve_dtype(dtype))

    def full(self, shape: Sequence[int], fill_value: Any, dtype: Any = None) -> mx.array:
        return mx.full(shape, h.fill_scalar(fill_value), dtype=self._resolve_dtype(dtype))

    def linspace(self, start: float, stop: float, num: int = 50) -> mx.array:
        return mx.linspace(start, stop, num, dtype=self._dtype())

    def arange(self, *args: Any, **kwargs: Any) -> mx.array:
        dtype = kwargs.pop("dtype", self._dtype())

        def _scalar(v: Any) -> int | float:
            if isinstance(v, mx.array):
                return float(v.item()) if v.ndim == 0 else float(v.reshape(-1)[0])
            if isinstance(v, np.generic):
                return v.item()
            return v

        args = tuple(_scalar(a) for a in args)
        if "step" in kwargs and kwargs["step"] is not None:
            kwargs["step"] = _scalar(kwargs["step"])
        return mx.arange(*args, dtype=dtype, **kwargs)

    def zeros_like(self, x: Any) -> mx.array:
        return mx.zeros_like(self.array(x))

    def ones_like(self, x: Any) -> mx.array:
        return mx.ones_like(self.array(x))

    def full_like(self, x: Any, fill_value: Any) -> mx.array:
        t = self.array(x)
        return mx.full(t.shape, h.fill_scalar(fill_value), dtype=t.dtype)

    def empty(self, shape: Sequence[int]) -> mx.array:
        return mx.zeros(shape, dtype=self._dtype())

    def empty_like(self, x: Any) -> mx.array:
        return mx.zeros_like(self.array(x))

    def eye(self, n: int) -> mx.array:
        return mx.eye(n, dtype=self._dtype())

    def asarray(self, x: Any, **kwargs: Any) -> mx.array:
        dtype = kwargs.pop("dtype", self._dtype())
        return mx.array(x, dtype=self._resolve_dtype(dtype))

    def cast(self, x: Any) -> mx.array:
        return self.array(x)

    def is_array_like(self, x: Any) -> bool:
        return isinstance(x, (mx.array, np.ndarray, list, tuple))

    def arange_indices(self, start: Any, stop: Any = None, step: int = 1) -> mx.array:
        if stop is None:
            stop = start
            start = 0
        return mx.arange(start, stop, step, dtype=mx.int64)

    def ravel(self, x: Any) -> mx.array:
        return self.array(x).reshape(-1)

    def transpose(self, x: Any, axes: Sequence[int] | None = None) -> mx.array:
        x = self.array(x)
        return mx.transpose(x, axes) if axes is not None else mx.transpose(x)

    def reshape(self, x: Any, shape: Sequence[int]) -> mx.array:
        return self.array(x).reshape(shape)

    def atleast_1d(self, x: Any) -> mx.array:
        t = self.array(x)
        return t.reshape(1) if t.ndim == 0 else t

    def atleast_2d(self, x: Any) -> mx.array:
        t = self.array(x)
        if t.ndim == 0:
            return t.reshape(1, 1)
        if t.ndim == 1:
            return t.reshape(1, -1)
        return t

    def as_array_1d(self, data: Any) -> mx.array:
        if isinstance(data, (int, float)):
            return self.array([data])
        if isinstance(data, (list, tuple)):
            return self.array(data)
        if self.is_array_like(data):
            return self.array(data).reshape(-1)
        raise ValueError("Unsupported type for as_array_1d")

    def stack(self, xs: Sequence[Any], axis: int = 0) -> mx.array:
        return mx.stack([self.cast(x) for x in xs], axis=axis)

    def concatenate(self, arrays: Sequence[Any], axis: int = 0) -> mx.array:
        return mx.concatenate([self.array(a) for a in arrays], axis=axis)

    def flip(self, x: Any) -> mx.array:
        return h.flip(self.array(x), axis=0)

    def roll(self, x: Any, shift: Any, axis: Any = ()) -> mx.array:
        ax = None if axis == () else axis
        return mx.roll(self.array(x), shift, axis=ax)

    def repeat(self, x: Any, repeats: int) -> mx.array:
        return mx.repeat(self.array(x), repeats)

    def broadcast_to(self, x: Any, shape: Sequence[int]) -> mx.array:
        return mx.broadcast_to(self.array(x), shape)

    def tile(self, x: Any, dims: Any) -> mx.array:
        d = dims if isinstance(dims, (list, tuple)) else (dims,)
        return mx.tile(self.array(x), d)

    def expand_dims(self, x: Any, axis: int) -> mx.array:
        return mx.expand_dims(self.array(x), axis)

    def meshgrid(self, *arrays: Any) -> tuple[mx.array, ...]:
        return mx.meshgrid(*[self.array(a) for a in arrays], indexing="xy")

    def unsqueeze_last(self, x: Any) -> mx.array:
        return mx.expand_dims(self.array(x), -1)

    def sum(self, x: Any, axis: int | None = None) -> mx.array:
        x = self.array(x)
        return mx.sum(x, axis=axis) if axis is not None else mx.sum(x)

    def mean(self, x: Any, axis: int | None = None, keepdims: bool = False) -> mx.array:
        return h.nanmean(self.array(x), axis=axis if not keepdims else axis)

    def std(self, x: Any, axis: int | None = None) -> mx.array:
        x = self.array(x)
        return mx.std(x, axis=axis) if axis is not None else mx.std(x)

    def max(self, x: Any) -> float:
        return float(mx.max(self.array(x)))

    def min(self, x: Any) -> float:
        return float(mx.min(self.array(x)))

    def argmin(self, x: Any, axis: int | None = None) -> mx.array:
        return mx.argmin(self.array(x), axis=axis)

    def argwhere(self, x: Any) -> mx.array:
        return h.argwhere(x)

    def clip(self, x: Any, a_min: Any, a_max: Any) -> mx.array:
        return mx.clip(self.array(x), a_min, a_max)

    def where(self, condition: Any, x: Any, y: Any) -> mx.array:
        if isinstance(condition, bool):
            return self.array(x if condition else y)
        return mx.where(condition, self.array(x), self.array(y))

    def maximum(self, a: Any, b: Any) -> mx.array:
        return mx.maximum(self.array(a), self.array(b))

    def minimum(self, a: Any, b: Any) -> mx.array:
        return mx.minimum(self.array(a), self.array(b))

    def fmax(self, a: Any, b: Any) -> mx.array:
        return h.fmax(a, b)

    def power(self, x: Any, y: Any) -> mx.array:
        return mx.power(self.array(x), self.array(y))

    def diff(self, x: Any, n: int = 1, axis: int = -1, **kwargs: Any) -> mx.array:
        return h.diff(x, n=n, axis=axis, **kwargs)

    def all(self, x: Any) -> bool:
        if isinstance(x, bool):
            return x
        return bool(mx.all(self.array(x)))

    def any(self, x: Any) -> bool:
        if isinstance(x, bool):
            return x
        return bool(mx.any(self.array(x)))

    def nanmax(self, x: Any, axis: int | None = None, keepdim: bool = False) -> mx.array:
        return h.nanmax(x, axis=axis, keepdims=keepdim)

    def sort(self, x: Any, axis: int = -1) -> mx.array:
        return mx.sort(self.array(x), axis=axis)

    def isclose(self, a: Any, b: Any, rtol: float = 1e-5, atol: float = 1e-8) -> mx.array:
        return mx.isclose(self.array(a), self.array(b), rtol=rtol, atol=atol)

    def allclose(self, a: Any, b: Any, rtol: float = 1e-5, atol: float = 1e-8) -> bool:
        return bool(mx.allclose(self.array(a), self.array(b), rtol=rtol, atol=atol))

    def radians(self, x: Any) -> mx.array:
        return h.deg2rad(x)

    def degrees(self, x: Any) -> mx.array:
        return h.rad2deg(x)

    def _u(self, x: Any) -> mx.array:
        return self.array(x)

    def sin(self, x: Any) -> mx.array:
        return mx.sin(self._u(x))

    def cos(self, x: Any) -> mx.array:
        return mx.cos(self._u(x))

    def tan(self, x: Any) -> mx.array:
        return mx.tan(self._u(x))

    def arcsin(self, x: Any) -> mx.array:
        return mx.arcsin(self._u(x))

    def arccos(self, x: Any) -> mx.array:
        return mx.arccos(self._u(x))

    def arctan(self, x: Any) -> mx.array:
        return mx.arctan(self._u(x))

    def arctan2(self, y: Any, x: Any) -> mx.array:
        return mx.arctan2(self._u(y), self._u(x))

    def sinh(self, x: Any) -> mx.array:
        return mx.sinh(self._u(x))

    def cosh(self, x: Any) -> mx.array:
        return mx.cosh(self._u(x))

    def tanh(self, x: Any) -> mx.array:
        return mx.tanh(self._u(x))

    def exp(self, x: Any) -> mx.array:
        x = self._u(x)
        if mx.issubdtype(x.dtype, mx.complexfloating):
            return mx.exp(x)
        return mx.exp(x)

    def log(self, x: Any) -> mx.array:
        return mx.log(self._u(x))

    def log2(self, x: Any) -> mx.array:
        return mx.log2(self._u(x))

    def log10(self, x: Any) -> mx.array:
        return mx.log10(self._u(x))

    def sqrt(self, x: Any) -> mx.array:
        return mx.sqrt(self._u(x))

    def abs(self, x: Any) -> mx.array:
        return mx.abs(self._u(x))

    def sign(self, x: Any) -> mx.array:
        return mx.sign(self._u(x))

    def floor(self, x: Any) -> mx.array:
        return mx.floor(self._u(x))

    def ceil(self, x: Any) -> mx.array:
        return mx.ceil(self._u(x))

    def hypot(self, x: Any, y: Any) -> mx.array:
        return h.hypot(x, y)

    def deg2rad(self, x: Any) -> mx.array:
        return h.deg2rad(x)

    def rad2deg(self, x: Any) -> mx.array:
        return h.rad2deg(x)

    def conj(self, x: Any) -> mx.array:
        return mx.conj(self._u(x))

    def real(self, x: Any) -> mx.array:
        return mx.real(self._u(x))

    def imag(self, x: Any) -> mx.array:
        return mx.imag(self._u(x))

    def copysign(self, x: Any, y: Any) -> mx.array:
        return h.copysign(x, y)

    def isnan(self, x: Any) -> Any:
        if isinstance(x, (int, float)):
            return np.isnan(x)
        return mx.isnan(self._u(x))

    def isinf(self, x: Any) -> Any:
        if isinstance(x, (int, float)):
            return np.isinf(x)
        return mx.isinf(self._u(x))

    def isfinite(self, x: Any) -> Any:
        if isinstance(x, (int, float)):
            return np.isfinite(x)
        return mx.isfinite(self._u(x))

    def matmul(self, a: Any, b: Any) -> mx.array:
        return mx.matmul(self.array(a), self.array(b))

    def cross(
        self,
        a: Any,
        b: Any,
        axisa: int = -1,
        axisb: int = -1,
        axisc: int = -1,
        axis: int | None = None,
    ) -> mx.array:
        if axis is not None:
            axisa = axisb = axisc = axis
        return mx.linalg.cross(self.array(a), self.array(b), axis=axisa)

    def batched_chain_matmul3(self, a: Any, b: Any, c: Any) -> mx.array:
        a, b, c = self.array(a), self.array(b), self.array(c)
        return mx.matmul(mx.matmul(a, b), c)

    def matrix_vector_multiply_and_squeeze(self, p: Any, E: Any) -> mx.array:
        p, e = self.array(p), self.array(E)
        return mx.squeeze(mx.matmul(p, mx.expand_dims(e, -1)), axis=-1)

    def mult_p_E(self, p: Any, E: Any) -> mx.array:
        p_c = self.array(p).astype(self._cdtype())
        e_c = self.array(E).astype(self._cdtype())
        return mx.squeeze(mx.matmul(p_c, mx.expand_dims(e_c, -1)), axis=-1)

    def lstsq(self, a: Any, b: Any) -> mx.array:
        return h.from_numpy(np.linalg.lstsq(h.to_numpy(a), h.to_numpy(b), rcond=None)[0])

    def to_complex(self, x: Any) -> mx.array:
        return self.array(x).astype(self._cdtype())

    def dot(self, a: Any, b: Any) -> mx.array:
        return h.dot(a, b)

    def outer(self, a: Any, b: Any) -> mx.array:
        return mx.outer(self.array(a), self.array(b))

    def einsum(self, subscripts: str, *operands: Any, **kwargs: Any) -> mx.array:
        return mx.einsum(subscripts, *[self.array(o) for o in operands], **kwargs)

    def interp(self, x: Any, xp: Any, fp: Any) -> mx.array:
        x = self.array(x)
        xp = self.array(xp)
        fp = self.array(fp)
        order = mx.argsort(xp)
        xp = xp[order]
        fp = fp[order]
        x_clip = mx.clip(x, xp[0], xp[-1])
        idx = h.searchsorted(xp, x_clip, side="right")
        idx = mx.clip(idx, 1, xp.shape[0] - 1)
        x0, x1 = xp[idx - 1], xp[idx]
        y0, y1 = fp[idx - 1], fp[idx]
        return y0 + (y1 - y0) * (x_clip - x0) / (x1 - x0)

    def nearest_nd_interpolator(self, points: Any, values: Any, Hx: Any, Hy: Any) -> mx.array:
        if Hx is None or Hy is None:
            raise ValueError("Hx and Hy must be provided")
        points = self.array(points)
        values = self.array(values)
        hx, hy = self.array(Hx), self.array(Hy)
        hx, hy = mx.broadcast_arrays(hx, hy)
        q_flat = mx.stack([hx, hy], axis=-1).reshape(-1, 2)
        d = mx.sqrt(mx.sum((q_flat[:, None, :] - points[None, :, :]) ** 2, axis=-1))
        idx = mx.argmin(d, axis=1)
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
    ) -> mx.array:
        inp = h.to_numpy(input)
        grd = h.to_numpy(grid)
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
        output = np.zeros((n, c, h_out, w_out), dtype=inp.dtype)
        order = 0 if mode == "nearest" else 1
        scipy_mode = "constant"
        if padding_mode == "border":
            scipy_mode = "nearest"
        elif padding_mode == "reflection":
            scipy_mode = "reflect"
        for batch in range(n):
            for ch in range(c):
                coords = np.stack((y_pix[batch], x_pix[batch]))
                output[batch, ch] = map_coordinates(
                    inp[batch, ch], coords, order=order, mode=scipy_mode, cval=0.0
                )
        return self.array(output)

    def get_bilinear_weights(self, coords: Any, bin_edges: Sequence[Any]) -> tuple[mx.array, mx.array]:
        x_edges, y_edges = self.array(bin_edges[0]), self.array(bin_edges[1])
        coords = self.array(coords)
        x, y = coords[:, 0], coords[:, 1]
        valid = (x >= x_edges[0]) & (x <= x_edges[-1]) & (y >= y_edges[0]) & (y <= y_edges[-1])
        x_centers = (x_edges[:-1] + x_edges[1:]) / 2
        y_centers = (y_edges[:-1] + y_edges[1:]) / 2
        ix = h.searchsorted(x_centers, x, side="right") - 1
        iy = h.searchsorted(y_centers, y, side="right") - 1
        ix = mx.clip(ix, 0, x_centers.shape[0] - 2)
        iy = mx.clip(iy, 0, y_centers.shape[0] - 2)
        x0, x1 = x_centers[ix], x_centers[ix + 1]
        y0, y1 = y_centers[iy], y_centers[iy + 1]
        wx = (x - x0) / (x1 - x0 + 1e-9)
        wy = (y - y0) / (y1 - y0 + 1e-9)
        w00 = (1 - wx) * (1 - wy)
        w01 = (1 - wx) * wy
        w10 = wx * (1 - wy)
        w11 = wx * wy
        all_indices = mx.stack(
            [
                mx.stack([ix, iy], axis=1),
                mx.stack([ix, iy + 1], axis=1),
                mx.stack([ix + 1, iy], axis=1),
                mx.stack([ix + 1, iy + 1], axis=1),
            ],
            axis=1,
        )
        all_weights = mx.stack([w00, w01, w10, w11], axis=1)
        all_weights = all_weights * valid[:, None].astype(all_weights.dtype)
        return all_indices, all_weights

    def polyfit(self, x: Any, y: Any, degree: int) -> mx.array:
        return h.from_numpy(np.polyfit(h.to_numpy(x), h.to_numpy(y), degree))

    def polyval(self, coeffs: Any, x: Any) -> mx.array:
        return h.from_numpy(np.polyval(h.to_numpy(coeffs), h.to_numpy(x)))

    def fftconvolve(
        self, in1: Any, in2: Any, mode: Literal["full", "valid", "same"] = "full"
    ) -> mx.array:
        a = h.to_numpy(in1)
        b = h.to_numpy(in2)
        return self.array(_scipy_fftconvolve(a, b, mode=mode))

    def default_rng(self, seed: int | None = None) -> Any:
        return mx.random.key(0 if seed is None else seed)

    def random_uniform(
        self, low: float = 0.0, high: float = 1.0, size: Any = None, generator: Any = None
    ) -> mx.array:
        key = generator if generator is not None else mx.random.key(0)
        key, sub = mx.random.split(key)
        shape = size or ()
        return low + (high - low) * mx.random.uniform(sub, shape=shape, dtype=self._dtype())

    def rand(self, *size: int) -> mx.array:
        shape = size or (1,)
        return mx.random.uniform(mx.random.key(0), shape=shape, dtype=self._dtype())

    def random_normal(
        self, loc: float = 0.0, scale: float = 1.0, size: Any = None, generator: Any = None
    ) -> mx.array:
        key = generator if generator is not None else mx.random.key(0)
        key, sub = mx.random.split(key)
        shape = size or ()
        return loc + scale * mx.random.normal(sub, shape=shape, dtype=self._dtype())

    def sobol_sampler(
        self, dim: int, num_samples: int, scramble: bool = True, seed: int | None = None
    ) -> mx.array:
        from scipy.stats import qmc

        n = 1 << (num_samples - 1).bit_length() if num_samples > 0 else num_samples
        sobol = qmc.Sobol(d=dim, scramble=scramble, seed=seed)
        samples = sobol.random(n=n)
        return self.array(samples[:num_samples].astype(np.float32))

    def erfinv(self, x: Any) -> mx.array:
        from scipy.special import erfinv

        return self.array(erfinv(h.to_numpy(x)))

    def factorial(self, n: Any) -> mx.array:
        from scipy.special import gamma

        return self.array(gamma(h.to_numpy(n) + 1))

    def path_contains_points(self, vertices: Any, points: Any) -> mx.array:
        from matplotlib.path import Path

        path = Path(h.to_numpy(vertices))
        mask = path.contains_points(h.to_numpy(points))
        return self.array(mask.astype(bool))

    def pad(
        self, tensor: Any, pad_width: Any, mode: str = "constant", constant_values: float | None = 0
    ) -> mx.array:
        if mode != "constant":
            raise NotImplementedError("Only constant mode supported")
        return mx.pad(self.array(tensor), pad_width, mode="constant", constant_values=constant_values)

    def vectorize(self, pyfunc: Callable[..., Any]) -> Callable[..., Any]:
        def wrapped(x: mx.array) -> mx.array:
            flat = x.reshape(-1)
            mapped = [pyfunc(float(v)) for v in flat]
            return mx.array(mapped).reshape(x.shape)

        return wrapped

    @contextlib.contextmanager
    def errstate(self, **kwargs: Any) -> Generator[None, None, None]:
        yield

    def histogram(self, x: Any, bins: Any = 10) -> tuple[mx.array, mx.array]:
        x_np = h.to_numpy(x)
        if isinstance(bins, int):
            counts, edges = np.histogram(x_np, bins=bins)
        else:
            counts, edges = np.histogram(x_np, bins=h.to_numpy(bins))
        return self.array(counts), self.array(edges)

    def histogram2d(
        self, x: Any, y: Any, bins: Any, weights: Any | None = None
    ) -> tuple[mx.array, mx.array, mx.array]:
        if not isinstance(bins, (list, tuple)) or len(bins) != 2:
            raise ValueError("`bins` must be a list or tuple of two edge arrays.")
        hist, xedges, yedges = np.histogram2d(
            h.to_numpy(x), h.to_numpy(y), bins=[h.to_numpy(bins[0]), h.to_numpy(bins[1])],
            weights=h.to_numpy(weights) if weights is not None else None,
        )
        return self.array(hist), self.array(xedges), self.array(yedges)

    def nansum(self, x: Any, axis: int | None = None) -> mx.array:
        return h.nansum(x, axis=axis)

    def nanmean(self, x: Any, axis: int | None = None) -> mx.array:
        return h.nanmean(x, axis=axis)

    def vstack(self, xs: Sequence[Any]) -> mx.array:
        return h.vstack(list(xs))

    def column_stack(self, xs: Sequence[Any]) -> mx.array:
        return h.column_stack(list(xs))

    def searchsorted(self, a: Any, v: Any, side: str = "left") -> mx.array:
        return h.searchsorted(a, v, side=side)

    def round(self, x: Any, decimals: int = 0) -> mx.array:
        return mx.round(self.array(x), decimals=decimals)

    def logical_and(self, a: Any, b: Any) -> mx.array:
        return mx.logical_and(self.array(a), self.array(b))

    def logical_or(self, a: Any, b: Any) -> mx.array:
        return mx.logical_or(self.array(a), self.array(b))

    def logical_not(self, x: Any) -> mx.array:
        return mx.logical_not(self.array(x))
