"""Small helpers bridging MLX arrays and NumPy/SciPy."""

from __future__ import annotations

from typing import Any

import mlx.core as mx
import numpy as np


def is_mx_array(x: Any) -> bool:
    return isinstance(x, mx.array)


def to_numpy(x: Any) -> np.ndarray:
    if isinstance(x, mx.array):
        return np.array(x)
    return np.asarray(x)


def from_numpy(x: np.ndarray, dtype: Any = None) -> mx.array:
    if dtype is not None:
        return mx.array(x, dtype=dtype)
    return mx.array(x)


def searchsorted(a: Any, v: Any, side: str = "left") -> mx.array:
    return mx.array(np.searchsorted(to_numpy(a), to_numpy(v), side=side))


def deg2rad(x: Any) -> mx.array:
    return mx.array(to_numpy(x)) * (np.pi / 180.0)


def rad2deg(x: Any) -> mx.array:
    return mx.array(to_numpy(x)) * (180.0 / np.pi)


def flip(x: Any, axis: int = 0) -> mx.array:
    return mx.array(np.flip(to_numpy(x), axis=axis))


def diff(x: Any, n: int = 1, axis: int = -1, **kwargs: Any) -> mx.array:
    return mx.array(np.diff(to_numpy(x), n=n, axis=axis, **kwargs))


def dot(a: Any, b: Any) -> mx.array:
    return mx.array(np.dot(to_numpy(a), to_numpy(b)))


def vstack(xs: list[Any]) -> mx.array:
    return mx.array(np.vstack([to_numpy(x) for x in xs]))


def column_stack(xs: list[Any]) -> mx.array:
    return mx.array(np.column_stack([to_numpy(x) for x in xs]))


def fmax(a: Any, b: Any) -> mx.array:
    return mx.array(np.fmax(to_numpy(a), to_numpy(b)))


def fmin(a: Any, b: Any) -> mx.array:
    return mx.array(np.fmin(to_numpy(a), to_numpy(b)))


def copysign(x: Any, y: Any) -> mx.array:
    return mx.array(np.copysign(to_numpy(x), to_numpy(y)))


def hypot(x: Any, y: Any) -> mx.array:
    return mx.array(np.hypot(to_numpy(x), to_numpy(y)))


def nansum(x: Any, axis: int | None = None) -> mx.array:
    return mx.array(np.nansum(to_numpy(x), axis=axis))


def nanmean(x: Any, axis: int | None = None) -> mx.array:
    return mx.array(np.nanmean(to_numpy(x), axis=axis))


def nanmax(x: Any, axis: int | None = None, keepdims: bool = False) -> mx.array:
    return mx.array(np.nanmax(to_numpy(x), axis=axis, keepdims=keepdims))


def argwhere(x: Any) -> mx.array:
    return mx.array(np.argwhere(to_numpy(x)))


def fill_scalar(fill_value: Any) -> Any:
    if isinstance(fill_value, mx.array):
        if fill_value.ndim == 0:
            return float(fill_value.item())
        if fill_value.size == 1:
            return float(fill_value.reshape(-1)[0])
    if isinstance(fill_value, np.ndarray) and fill_value.size == 1:
        return float(fill_value.reshape(-1)[0])
    return fill_value
