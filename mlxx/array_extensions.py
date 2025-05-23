import mlx.core as mx

def _array_tolist(self):
    """
    Recursively convert an MLX array to a (nested) Python list using MLX ops only.
    """
    # Scalar case
    if self.shape == ():
        return self.item()
    # 1D
    elif len(self.shape) == 1:
        return [self[i].item() for i in range(self.shape[0])]
    # ND
    else:
        return [mx.array(self[i]).tolist() for i in range(self.shape[0])]

if not hasattr(mx.array, "tolist"):
    mx.array.tolist = _array_tolist


def _array_allclose(self, b, rtol=1e-05, atol=1e-08, equal_nan=False, stream=None):
    """
    Internal wrapper for mx.allclose.
    """
    return mx.allclose(
        self, b, rtol=rtol, atol=atol, equal_nan=equal_nan, stream=stream
    )


if not hasattr(mx.array, "allclose"):
    mx.array.allclose = _array_allclose


def _array_isclose(self, b, rtol=1e-05, atol=1e-08, equal_nan=False, stream=None):
    """
    Internal wrapper for mx.isclose.
    """
    return mx.isclose(self, b, rtol=rtol, atol=atol, equal_nan=equal_nan, stream=stream)


if not hasattr(mx.array, "isclose"):
    mx.array.isclose = _array_isclose


def _array_array_equal(self, b, equal_nan=False, stream=None):
    """
    Internal wrapper for mx.array_equal.
    """
    return mx.array_equal(self, b, equal_nan=equal_nan, stream=stream)


if not hasattr(mx.array, "array_equal"):
    mx.array.array_equal = _array_array_equal


def _array_logical_and(self, b, stream=None):
    """
    Internal wrapper for mx.logical_and.
    """
    return mx.logical_and(self, b, stream=stream)


if not hasattr(mx.array, "logical_and"):
    mx.array.logical_and = _array_logical_and


def _array_logical_or(self, b, stream=None):
    """
    Internal wrapper for mx.logical_or.
    """
    return mx.logical_or(self, b, stream=stream)


if not hasattr(mx.array, "logical_or"):
    mx.array.logical_or = _array_logical_or


def _array_logical_xor(self, b, stream=None):
    """
    Internal wrapper for mx.logical_xor.
    """
    return mx.logical_xor(self, b, stream=stream)


if not hasattr(mx.array, "logical_xor"):
    mx.array.logical_xor = _array_logical_xor


def _array_binary_maximum(self, b, stream=None):
    """
    Internal wrapper for mx.maximum.
    """
    return mx.maximum(self, b, stream=stream)


if not hasattr(mx.array, "binary_maximum"):
    mx.array.binary_maximum = _array_binary_maximum


def _array_binary_minimum(self, b, stream=None):
    """
    Internal wrapper for mx.minimum.
    """
    return mx.minimum(self, b, stream=stream)


if not hasattr(mx.array, "binary_minimum"):
    mx.array.binary_minimum = _array_binary_minimum


def _array_power(self, exponent, stream=None):
    """
    Internal wrapper for mx.power.
    """
    return mx.power(self, exponent, stream=stream)


if not hasattr(mx.array, "power"):
    mx.array.power = _array_power


def _array_matmul(self, b, stream=None):
    """
    Internal wrapper for mx.matmul.
    """
    return mx.matmul(self, b, stream=stream)


if not hasattr(mx.array, "matmul"):
    mx.array.matmul = _array_matmul


def _array_inner(self, b, stream=None):
    """
    Internal wrapper for mx.inner.
    """
    return mx.inner(self, b, stream=stream)


if not hasattr(mx.array, "inner"):
    mx.array.inner = _array_inner

# Unary Operations from mlx.core


def _array_arccos(self, stream=None):
    """Internal wrapper for mx.arccos."""
    return mx.arccos(self, stream=stream)


if not hasattr(mx.array, "arccos"):
    mx.array.arccos = _array_arccos


def _array_arccosh(self, stream=None):
    """Internal wrapper for mx.arccosh."""
    return mx.arccosh(self, stream=stream)


if not hasattr(mx.array, "arccosh"):
    mx.array.arccosh = _array_arccosh


def _array_arcsin(self, stream=None):
    """Internal wrapper for mx.arcsin."""
    return mx.arcsin(self, stream=stream)


if not hasattr(mx.array, "arcsin"):
    mx.array.arcsin = _array_arcsin


def _array_arcsinh(self, stream=None):
    """Internal wrapper for mx.arcsinh."""
    return mx.arcsinh(self, stream=stream)


if not hasattr(mx.array, "arcsinh"):
    mx.array.arcsinh = _array_arcsinh


def _array_arctan(self, stream=None):
    """Internal wrapper for mx.arctan."""
    return mx.arctan(self, stream=stream)


if not hasattr(mx.array, "arctan"):
    mx.array.arctan = _array_arctan


def _array_arctanh(self, stream=None):
    """Internal wrapper for mx.arctanh."""
    return mx.arctanh(self, stream=stream)


if not hasattr(mx.array, "arctanh"):
    mx.array.arctanh = _array_arctanh


def _array_ceil(self, stream=None):
    """Internal wrapper for mx.ceil."""
    return mx.ceil(self, stream=stream)


if not hasattr(mx.array, "ceil"):
    mx.array.ceil = _array_ceil


def _array_cosh(self, stream=None):
    """Internal wrapper for mx.cosh."""
    return mx.cosh(self, stream=stream)


if not hasattr(mx.array, "cosh"):
    mx.array.cosh = _array_cosh


def _array_degrees(self, stream=None):
    """Internal wrapper for mx.degrees."""
    return mx.degrees(self, stream=stream)


if not hasattr(mx.array, "degrees"):
    mx.array.degrees = _array_degrees


def _array_erf(self, stream=None):
    """Internal wrapper for mx.erf."""
    return mx.erf(self, stream=stream)


if not hasattr(mx.array, "erf"):
    mx.array.erf = _array_erf


def _array_erfinv(self, stream=None):
    """Internal wrapper for mx.erfinv."""
    return mx.erfinv(self, stream=stream)


if not hasattr(mx.array, "erfinv"):
    mx.array.erfinv = _array_erfinv


def _array_expm1(self, stream=None):
    """Internal wrapper for mx.expm1."""
    return mx.expm1(self, stream=stream)


if not hasattr(mx.array, "expm1"):
    mx.array.expm1 = _array_expm1


def _array_floor(self, stream=None):
    """Internal wrapper for mx.floor."""
    return mx.floor(self, stream=stream)


if not hasattr(mx.array, "floor"):
    mx.array.floor = _array_floor


def _array_imag(self, stream=None):
    """Internal wrapper for mx.imag."""
    return mx.imag(self, stream=stream)


if not hasattr(mx.array, "imag"):
    mx.array.imag = _array_imag


def _array_isfinite(self, stream=None):
    """Internal wrapper for mx.isfinite."""
    return mx.isfinite(self, stream=stream)


if not hasattr(mx.array, "isfinite"):
    mx.array.isfinite = _array_isfinite


def _array_isinf(self, stream=None):
    """Internal wrapper for mx.isinf."""
    return mx.isinf(self, stream=stream)


if not hasattr(mx.array, "isinf"):
    mx.array.isinf = _array_isinf


def _array_isnan(self, stream=None):
    """Internal wrapper for mx.isnan."""
    return mx.isnan(self, stream=stream)


if not hasattr(mx.array, "isnan"):
    mx.array.isnan = _array_isnan


def _array_isneginf(self, stream=None):
    """Internal wrapper for mx.isneginf."""
    return mx.isneginf(self, stream=stream)


if not hasattr(mx.array, "isneginf"):
    mx.array.isneginf = _array_isneginf


def _array_isposinf(self, stream=None):
    """Internal wrapper for mx.isposinf."""
    return mx.isposinf(self, stream=stream)


if not hasattr(mx.array, "isposinf"):
    mx.array.isposinf = _array_isposinf


def _array_logical_not(self, stream=None):
    """Internal wrapper for mx.logical_not."""
    return mx.logical_not(self, stream=stream)


if not hasattr(mx.array, "logical_not"):
    mx.array.logical_not = _array_logical_not


def _array_negative(self, stream=None):
    """Internal wrapper for mx.negative."""
    return mx.negative(self, stream=stream)


if not hasattr(mx.array, "negative"):
    mx.array.negative = _array_negative


def _array_radians(self, stream=None):
    """Internal wrapper for mx.radians."""
    return mx.radians(self, stream=stream)


if not hasattr(mx.array, "radians"):
    mx.array.radians = _array_radians


def _array_real(self, stream=None):
    """Internal wrapper for mx.real."""
    return mx.real(self, stream=stream)


if not hasattr(mx.array, "real"):
    mx.array.real = _array_real


def _array_sigmoid(self, stream=None):
    """Internal wrapper for mx.sigmoid."""
    return mx.sigmoid(self, stream=stream)


if not hasattr(mx.array, "sigmoid"):
    mx.array.sigmoid = _array_sigmoid


def _array_sign(self, stream=None):
    """Internal wrapper for mx.sign."""
    return mx.sign(self, stream=stream)


if not hasattr(mx.array, "sign"):
    mx.array.sign = _array_sign


def _array_sinh(self, stream=None):
    """Internal wrapper for mx.sinh."""
    return mx.sinh(self, stream=stream)


if not hasattr(mx.array, "sinh"):
    mx.array.sinh = _array_sinh


def _array_tan(self, stream=None):
    """Internal wrapper for mx.tan."""
    return mx.tan(self, stream=stream)


if not hasattr(mx.array, "tan"):
    mx.array.tan = _array_tan


def _array_tanh(self, stream=None):
    """Internal wrapper for mx.tanh."""
    return mx.tanh(self, stream=stream)


if not hasattr(mx.array, "tanh"):
    mx.array.tanh = _array_tanh


def _array_stop_gradient(self, stream=None):
    """Internal wrapper for mx.stop_gradient."""
    return mx.stop_gradient(self, stream=stream)


if not hasattr(mx.array, "stop_gradient"):
    mx.array.stop_gradient = _array_stop_gradient


# some convenient methods inspired by PyTorch

if not hasattr(mx.array, "permute"):
    mx.array.permute = mx.array.transpose

def _array_abs(self, stream=None):
    """Internal wrapper for mx.abs."""
    return mx.abs(self, stream=stream)

if not hasattr(mx.array, "abs"):
    mx.array.abs = _array_abs

def _array_prod(self, axis=None, keepdims=False, stream=None):
    """
    Internal wrapper for mx.prod.
    """
    return mx.prod(self, axis=axis, keepdims=keepdims, stream=stream)

if not hasattr(mx.array, "prod"):
    mx.array.prod = _array_prod

def _array_any(self, axis=None, keepdims=False, stream=None):
    """
    Internal wrapper for mx.any.
    """
    return mx.any(self, axis=axis, keepdims=keepdims, stream=stream)

if not hasattr(mx.array, "any"):
    mx.array.any = _array_any

def _array_clone(self, stream=None):
    """
    Returns a copy of the array (deep copy).
    """
    return mx.array(self, stream=stream)

if not hasattr(mx.array, "clone"):
    mx.array.clone = _array_clone

def _array_reshape(self, shape, stream=None):
    """
    Internal wrapper for mx.reshape.
    """
    return mx.reshape(self, shape, stream=stream)

if not hasattr(mx.array, "reshape"):
    mx.array.reshape = _array_reshape

def _array_view(self, *args, **kwargs):
    """
    Alias for reshape.
    """
    return self.reshape(*args, **kwargs)

if not hasattr(mx.array, "view"):
    mx.array.view = _array_view

def _array_unsqueeze(self, axis):
    """
    Insert a new axis at the specified position.
    """
    return mx.expand_dims(self, axis)

if not hasattr(mx.array, "unsqueeze"):
    mx.array.unsqueeze = _array_unsqueeze

def _array_eq(self, other):
    """
    Elementwise equality with another array or scalar.
    """
    return mx.equal(self, other)

if not hasattr(mx.array, "eq"):
    mx.array.eq = _array_eq

def _array_ne(self, other):
    """
    Elementwise not-equal with another array or scalar.
    """
    return mx.not_equal(self, other)

if not hasattr(mx.array, "ne"):
    mx.array.ne = _array_ne

def _array_gt(self, other):
    """
    Elementwise greater-than with another array or scalar.
    """
    return mx.greater(self, other)

if not hasattr(mx.array, "gt"):
    mx.array.gt = _array_gt
