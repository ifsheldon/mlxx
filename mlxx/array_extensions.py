import mlx.core as mx
from fastcore.basics import patch


@patch
def allclose(self: mx.array, *args, **kwargs):
    """
    Internal wrapper for mx.allclose.
    """
    return mx.allclose(self, *args, **kwargs)


@patch
def isclose(self: mx.array, *args, **kwargs):
    """
    Internal wrapper for mx.isclose.
    """
    return mx.isclose(self, *args, **kwargs)


@patch
def array_equal(self: mx.array, *args, **kwargs):
    """
    Internal wrapper for mx.array_equal.
    """
    return mx.array_equal(self, *args, **kwargs)


@patch
def logical_and(self: mx.array, *args, **kwargs):
    """
    Internal wrapper for mx.logical_and.
    """
    return mx.logical_and(self, *args, **kwargs)


@patch
def logical_or(self: mx.array, *args, **kwargs):
    """
    Internal wrapper for mx.logical_or.
    """
    return mx.logical_or(self, *args, **kwargs)


@patch
def binary_maximum(self: mx.array, *args, **kwargs):
    """
    Internal wrapper for mx.maximum.
    """
    return mx.maximum(self, *args, **kwargs)


@patch
def binary_minimum(self: mx.array, *args, **kwargs):
    """
    Internal wrapper for mx.minimum.
    """
    return mx.minimum(self, *args, **kwargs)


@patch
def power(self: mx.array, *args, **kwargs):
    """
    Internal wrapper for mx.power.
    """
    return mx.power(self, *args, **kwargs)


@patch
def matmul(self: mx.array, *args, **kwargs):
    """
    Internal wrapper for mx.matmul.
    """
    return mx.matmul(self, *args, **kwargs)


@patch
def inner(self: mx.array, *args, **kwargs):
    """
    Internal wrapper for mx.inner.
    """
    return mx.inner(self, *args, **kwargs)


# Unary Operations from mlx.core


@patch
def arccos(self: mx.array, *args, **kwargs):
    """Internal wrapper for mx.arccos."""
    return mx.arccos(self, *args, **kwargs)


@patch
def arccosh(self: mx.array, *args, **kwargs):
    """Internal wrapper for mx.arccosh."""
    return mx.arccosh(self, *args, **kwargs)


@patch
def arcsin(self: mx.array, *args, **kwargs):
    """Internal wrapper for mx.arcsin."""
    return mx.arcsin(self, *args, **kwargs)


@patch
def arcsinh(self: mx.array, *args, **kwargs):
    """Internal wrapper for mx.arcsinh."""
    return mx.arcsinh(self, *args, **kwargs)


@patch
def arctan(self: mx.array, *args, **kwargs):
    """Internal wrapper for mx.arctan."""
    return mx.arctan(self, *args, **kwargs)


@patch
def arctanh(self: mx.array, *args, **kwargs):
    """Internal wrapper for mx.arctanh."""
    return mx.arctanh(self, *args, **kwargs)


@patch
def ceil(self: mx.array, *args, **kwargs):
    """Internal wrapper for mx.ceil."""
    return mx.ceil(self, *args, **kwargs)


@patch
def cosh(self: mx.array, *args, **kwargs):
    """Internal wrapper for mx.cosh."""
    return mx.cosh(self, *args, **kwargs)


@patch
def degrees(self: mx.array, *args, **kwargs):
    """Internal wrapper for mx.degrees."""
    return mx.degrees(self, *args, **kwargs)


@patch
def erf(self: mx.array, *args, **kwargs):
    """Internal wrapper for mx.erf."""
    return mx.erf(self, *args, **kwargs)


@patch
def erfinv(self: mx.array, *args, **kwargs):
    """Internal wrapper for mx.erfinv."""
    return mx.erfinv(self, *args, **kwargs)


@patch
def expm1(self: mx.array, *args, **kwargs):
    """Internal wrapper for mx.expm1."""
    return mx.expm1(self, *args, **kwargs)


@patch
def floor(self: mx.array, *args, **kwargs):
    """Internal wrapper for mx.floor."""
    return mx.floor(self, *args, **kwargs)


@patch
def isfinite(self: mx.array, *args, **kwargs):
    """Internal wrapper for mx.isfinite."""
    return mx.isfinite(self, *args, **kwargs)


@patch
def isinf(self: mx.array, *args, **kwargs):
    """Internal wrapper for mx.isinf."""
    return mx.isinf(self, *args, **kwargs)


@patch
def isnan(self: mx.array, *args, **kwargs):
    """Internal wrapper for mx.isnan."""
    return mx.isnan(self, *args, **kwargs)


@patch
def isneginf(self: mx.array, *args, **kwargs):
    """Internal wrapper for mx.isneginf."""
    return mx.isneginf(self, *args, **kwargs)


@patch
def isposinf(self: mx.array, *args, **kwargs):
    """Internal wrapper for mx.isposinf."""
    return mx.isposinf(self, *args, **kwargs)


@patch
def logical_not(self: mx.array, *args, **kwargs):
    """Internal wrapper for mx.logical_not."""
    return mx.logical_not(self, *args, **kwargs)


@patch
def negative(self: mx.array, *args, **kwargs):
    """Internal wrapper for mx.negative."""
    return mx.negative(self, *args, **kwargs)


@patch
def radians(self: mx.array, *args, **kwargs):
    """Internal wrapper for mx.radians."""
    return mx.radians(self, *args, **kwargs)


@patch
def sigmoid(self: mx.array, *args, **kwargs):
    """Internal wrapper for mx.sigmoid."""
    return mx.sigmoid(self, *args, **kwargs)


@patch
def sign(self: mx.array, *args, **kwargs):
    """Internal wrapper for mx.sign."""
    return mx.sign(self, *args, **kwargs)


@patch
def sinh(self: mx.array, *args, **kwargs):
    """Internal wrapper for mx.sinh."""
    return mx.sinh(self, *args, **kwargs)


@patch
def tan(self: mx.array, *args, **kwargs):
    """Internal wrapper for mx.tan."""
    return mx.tan(self, *args, **kwargs)


@patch
def tanh(self: mx.array, *args, **kwargs):
    """Internal wrapper for mx.tanh."""
    return mx.tanh(self, *args, **kwargs)


@patch
def stop_gradient(self: mx.array, *args, **kwargs):
    """Internal wrapper for mx.stop_gradient."""
    return mx.stop_gradient(self, *args, **kwargs)


# some convenient methods inspired by PyTorch

if not hasattr(mx.array, "permute"):
    mx.array.permute = mx.array.transpose

if not hasattr(mx.array, "t"):
    mx.array.t = mx.array.transpose


@patch
def add(self: mx.array, *args, **kwargs):
    """Internal wrapper for mx.add."""
    return mx.add(self, *args, **kwargs)


@patch
def addmm(self: mx.array, *args, **kwargs):
    """Internal wrapper for mx.addmm."""
    return mx.addmm(self, *args, **kwargs)


@patch
def logaddexp(self: mx.array, *args, **kwargs):
    """Internal wrapper for mx.logaddexp."""
    return mx.logaddexp(self, *args, **kwargs)


@patch
def multiply(self: mx.array, *args, **kwargs):
    """Internal wrapper for mx.multiply."""
    return mx.multiply(self, *args, **kwargs)


if not hasattr(mx.array, "mul"):
    mx.array.mul = mx.array.multiply  # alias


@patch
def nansum(self: mx.array, axis=None, keepdims=False, dtype=None, stream=None):
    """Mimics np.nansum and torch.nansum behavior for an array.
    Treats NaNs as zero.
    """
    if self.dtype == mx.bool_:
        # For boolean arrays, isnan is not directly applicable in the same way.
        # nansum on a boolean array usually means sum of True values.
        # If we need to handle potential NaNs that got into a bool array somehow (e.g. via view),
        # this might need specific handling. Assuming typical bool array usage.
        return mx.sum(self, axis=axis, keepdims=keepdims, stream=stream)

    # Replace NaNs with zeros
    # Ensure the 0.0 is of the same dtype as the array to avoid type promotion issues.
    zeros = mx.array(0.0, dtype=self.dtype)
    arr_without_nans = mx.where(
        mx.isnan(self, stream=stream), zeros, self, stream=stream
    )

    # Sum the array with NaNs replaced by zeros
    result = mx.sum(arr_without_nans, axis=axis, keepdims=keepdims, stream=stream)

    # Cast to specified dtype if provided
    if dtype is not None:
        result = result.astype(dtype, stream=stream)

    return result


@patch
def divide(self: mx.array, *args, **kwargs):
    """Internal wrapper for mx.divide."""
    return mx.divide(self, *args, **kwargs)


if not hasattr(mx.array, "div"):
    mx.array.div = mx.array.divide  # alias


@patch
def norm(self: mx.array, *args, **kwargs):
    """Internal wrapper for mx.linalg.norm."""
    return mx.linalg.norm(self, *args, **kwargs)


@patch
def numpy(self: mx.array, copy: bool = True):
    """Internal wrapper for numpy.array."""
    import numpy as np

    return np.array(self, copy=copy)
