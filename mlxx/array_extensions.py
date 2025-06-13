import mlx.core as mx


def _array_allclose(self, *args, **kwargs):
    """
    Internal wrapper for mx.allclose.
    """
    return mx.allclose(self, *args, **kwargs)


if not hasattr(mx.array, "allclose"):
    mx.array.allclose = _array_allclose


def _array_isclose(self, *args, **kwargs):
    """
    Internal wrapper for mx.isclose.
    """
    return mx.isclose(self, *args, **kwargs)


if not hasattr(mx.array, "isclose"):
    mx.array.isclose = _array_isclose


def _array_array_equal(self, *args, **kwargs):
    """
    Internal wrapper for mx.array_equal.
    """
    return mx.array_equal(self, *args, **kwargs)


if not hasattr(mx.array, "array_equal"):
    mx.array.array_equal = _array_array_equal


def _array_logical_and(self, *args, **kwargs):
    """
    Internal wrapper for mx.logical_and.
    """
    return mx.logical_and(self, *args, **kwargs)


if not hasattr(mx.array, "logical_and"):
    mx.array.logical_and = _array_logical_and


def _array_logical_or(self, *args, **kwargs):
    """
    Internal wrapper for mx.logical_or.
    """
    return mx.logical_or(self, *args, **kwargs)


if not hasattr(mx.array, "logical_or"):
    mx.array.logical_or = _array_logical_or


def _array_binary_maximum(self, *args, **kwargs):
    """
    Internal wrapper for mx.maximum.
    """
    return mx.maximum(self, *args, **kwargs)


if not hasattr(mx.array, "binary_maximum"):
    mx.array.binary_maximum = _array_binary_maximum


def _array_binary_minimum(self, *args, **kwargs):
    """
    Internal wrapper for mx.minimum.
    """
    return mx.minimum(self, *args, **kwargs)


if not hasattr(mx.array, "binary_minimum"):
    mx.array.binary_minimum = _array_binary_minimum


def _array_power(self, *args, **kwargs):
    """
    Internal wrapper for mx.power.
    """
    return mx.power(self, *args, **kwargs)


if not hasattr(mx.array, "power"):
    mx.array.power = _array_power


def _array_matmul(self, *args, **kwargs):
    """
    Internal wrapper for mx.matmul.
    """
    return mx.matmul(self, *args, **kwargs)


if not hasattr(mx.array, "matmul"):
    mx.array.matmul = _array_matmul


def _array_inner(self, *args, **kwargs):
    """
    Internal wrapper for mx.inner.
    """
    return mx.inner(self, *args, **kwargs)


if not hasattr(mx.array, "inner"):
    mx.array.inner = _array_inner

# Unary Operations from mlx.core


def _array_arccos(self, *args, **kwargs):
    """Internal wrapper for mx.arccos."""
    return mx.arccos(self, *args, **kwargs)


if not hasattr(mx.array, "arccos"):
    mx.array.arccos = _array_arccos


def _array_arccosh(self, *args, **kwargs):
    """Internal wrapper for mx.arccosh."""
    return mx.arccosh(self, *args, **kwargs)


if not hasattr(mx.array, "arccosh"):
    mx.array.arccosh = _array_arccosh


def _array_arcsin(self, *args, **kwargs):
    """Internal wrapper for mx.arcsin."""
    return mx.arcsin(self, *args, **kwargs)


if not hasattr(mx.array, "arcsin"):
    mx.array.arcsin = _array_arcsin


def _array_arcsinh(self, *args, **kwargs):
    """Internal wrapper for mx.arcsinh."""
    return mx.arcsinh(self, *args, **kwargs)


if not hasattr(mx.array, "arcsinh"):
    mx.array.arcsinh = _array_arcsinh


def _array_arctan(self, *args, **kwargs):
    """Internal wrapper for mx.arctan."""
    return mx.arctan(self, *args, **kwargs)


if not hasattr(mx.array, "arctan"):
    mx.array.arctan = _array_arctan


def _array_arctanh(self, *args, **kwargs):
    """Internal wrapper for mx.arctanh."""
    return mx.arctanh(self, *args, **kwargs)


if not hasattr(mx.array, "arctanh"):
    mx.array.arctanh = _array_arctanh


def _array_ceil(self, *args, **kwargs):
    """Internal wrapper for mx.ceil."""
    return mx.ceil(self, *args, **kwargs)


if not hasattr(mx.array, "ceil"):
    mx.array.ceil = _array_ceil


def _array_cosh(self, *args, **kwargs):
    """Internal wrapper for mx.cosh."""
    return mx.cosh(self, *args, **kwargs)


if not hasattr(mx.array, "cosh"):
    mx.array.cosh = _array_cosh


def _array_degrees(self, *args, **kwargs):
    """Internal wrapper for mx.degrees."""
    return mx.degrees(self, *args, **kwargs)


if not hasattr(mx.array, "degrees"):
    mx.array.degrees = _array_degrees


def _array_erf(self, *args, **kwargs):
    """Internal wrapper for mx.erf."""
    return mx.erf(self, *args, **kwargs)


if not hasattr(mx.array, "erf"):
    mx.array.erf = _array_erf


def _array_erfinv(self, *args, **kwargs):
    """Internal wrapper for mx.erfinv."""
    return mx.erfinv(self, *args, **kwargs)


if not hasattr(mx.array, "erfinv"):
    mx.array.erfinv = _array_erfinv


def _array_expm1(self, *args, **kwargs):
    """Internal wrapper for mx.expm1."""
    return mx.expm1(self, *args, **kwargs)


if not hasattr(mx.array, "expm1"):
    mx.array.expm1 = _array_expm1


def _array_floor(self, *args, **kwargs):
    """Internal wrapper for mx.floor."""
    return mx.floor(self, *args, **kwargs)


if not hasattr(mx.array, "floor"):
    mx.array.floor = _array_floor


def _array_isfinite(self, *args, **kwargs):
    """Internal wrapper for mx.isfinite."""
    return mx.isfinite(self, *args, **kwargs)


if not hasattr(mx.array, "isfinite"):
    mx.array.isfinite = _array_isfinite


def _array_isinf(self, *args, **kwargs):
    """Internal wrapper for mx.isinf."""
    return mx.isinf(self, *args, **kwargs)


if not hasattr(mx.array, "isinf"):
    mx.array.isinf = _array_isinf


def _array_isnan(self, *args, **kwargs):
    """Internal wrapper for mx.isnan."""
    return mx.isnan(self, *args, **kwargs)


if not hasattr(mx.array, "isnan"):
    mx.array.isnan = _array_isnan


def _array_isneginf(self, *args, **kwargs):
    """Internal wrapper for mx.isneginf."""
    return mx.isneginf(self, *args, **kwargs)


if not hasattr(mx.array, "isneginf"):
    mx.array.isneginf = _array_isneginf


def _array_isposinf(self, *args, **kwargs):
    """Internal wrapper for mx.isposinf."""
    return mx.isposinf(self, *args, **kwargs)


if not hasattr(mx.array, "isposinf"):
    mx.array.isposinf = _array_isposinf


def _array_logical_not(self, *args, **kwargs):
    """Internal wrapper for mx.logical_not."""
    return mx.logical_not(self, *args, **kwargs)


if not hasattr(mx.array, "logical_not"):
    mx.array.logical_not = _array_logical_not


def _array_negative(self, *args, **kwargs):
    """Internal wrapper for mx.negative."""
    return mx.negative(self, *args, **kwargs)


if not hasattr(mx.array, "negative"):
    mx.array.negative = _array_negative


def _array_radians(self, *args, **kwargs):
    """Internal wrapper for mx.radians."""
    return mx.radians(self, *args, **kwargs)


if not hasattr(mx.array, "radians"):
    mx.array.radians = _array_radians


def _array_sigmoid(self, *args, **kwargs):
    """Internal wrapper for mx.sigmoid."""
    return mx.sigmoid(self, *args, **kwargs)


if not hasattr(mx.array, "sigmoid"):
    mx.array.sigmoid = _array_sigmoid


def _array_sign(self, *args, **kwargs):
    """Internal wrapper for mx.sign."""
    return mx.sign(self, *args, **kwargs)


if not hasattr(mx.array, "sign"):
    mx.array.sign = _array_sign


def _array_sinh(self, *args, **kwargs):
    """Internal wrapper for mx.sinh."""
    return mx.sinh(self, *args, **kwargs)


if not hasattr(mx.array, "sinh"):
    mx.array.sinh = _array_sinh


def _array_tan(self, *args, **kwargs):
    """Internal wrapper for mx.tan."""
    return mx.tan(self, *args, **kwargs)


if not hasattr(mx.array, "tan"):
    mx.array.tan = _array_tan


def _array_tanh(self, *args, **kwargs):
    """Internal wrapper for mx.tanh."""
    return mx.tanh(self, *args, **kwargs)


if not hasattr(mx.array, "tanh"):
    mx.array.tanh = _array_tanh


def _array_stop_gradient(self, *args, **kwargs):
    """Internal wrapper for mx.stop_gradient."""
    return mx.stop_gradient(self, *args, **kwargs)


if not hasattr(mx.array, "stop_gradient"):
    mx.array.stop_gradient = _array_stop_gradient


# some convenient methods inspired by PyTorch

if not hasattr(mx.array, "permute"):
    mx.array.permute = mx.array.transpose

if not hasattr(mx.array, "t"):
    mx.array.t = mx.array.transpose


def _array_add(self, *args, **kwargs):
    """Internal wrapper for mx.add."""
    return mx.add(self, *args, **kwargs)


if not hasattr(mx.array, "add"):
    mx.array.add = _array_add


def _array_addmm(self, *args, **kwargs):
    """Internal wrapper for mx.addmm."""
    return mx.addmm(self, *args, **kwargs)


if not hasattr(mx.array, "addmm"):
    mx.array.addmm = _array_addmm


def _array_logaddexp(self, *args, **kwargs):
    """Internal wrapper for mx.logaddexp."""
    return mx.logaddexp(self, *args, **kwargs)


if not hasattr(mx.array, "logaddexp"):
    mx.array.logaddexp = _array_logaddexp


def _array_multiply(self, *args, **kwargs):
    """Internal wrapper for mx.multiply."""
    return mx.multiply(self, *args, **kwargs)


if not hasattr(mx.array, "multiply"):
    mx.array.multiply = _array_multiply


if not hasattr(mx.array, "mul"):
    mx.array.mul = _array_multiply  # alias


def _array_nansum(self, axis=None, keepdims=False, dtype=None, stream=None):
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


if not hasattr(mx.array, "nansum"):
    mx.array.nansum = _array_nansum


def _array_divide(self, *args, **kwargs):
    """Internal wrapper for mx.divide."""
    return mx.divide(self, *args, **kwargs)


if not hasattr(mx.array, "divide"):
    mx.array.divide = _array_divide

if not hasattr(mx.array, "div"):
    mx.array.div = _array_divide  # alias


def _array_norm(self, *args, **kwargs):
    """Internal wrapper for mx.linalg.norm."""
    return mx.linalg.norm(self, *args, **kwargs)


if not hasattr(mx.array, "norm"):
    mx.array.norm = _array_norm
