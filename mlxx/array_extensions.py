import mlx.core as mx


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

# ========== Additional PyTorch/NumPy-like methods for mlx.core.array ==========

# ----------- item -----------
def _array_item(self):
    """
    Return the Python scalar value if the array is scalar (size == 1), else raise ValueError.
    """
    if hasattr(self, "shape"):
        shape = self.shape
    else:
        shape = mx.shape(self)
    from functools import reduce
    import operator
    size = reduce(operator.mul, shape, 1)
    if size != 1:
        raise ValueError("item() can only be called on scalar arrays (size == 1), got shape {}".format(shape))
    # Try to get the value as a Python scalar
    # Use .tolist() should give a scalar
    value = self.tolist()
    if isinstance(value, (list, tuple)):
        # Defensive: flatten recursively
        def flatten(x):
            while isinstance(x, (list, tuple)):
                if len(x) != 1:
                    raise ValueError("item() can only be called on scalar arrays (size == 1), got shape {}".format(shape))
                x = x[0]
            return x
        value = flatten(value)
    return value

if not hasattr(mx.array, "item"):
    mx.array.item = _array_item

# ----------- mean -----------
def _array_mean(self, axis=None, keepdims=False):
    """
    Return the mean of the array elements over the given axis.
    """
    if hasattr(mx, "mean"):
        return mx.mean(self, axis=axis, keepdims=keepdims)
    else:
        # fallback: sum divided by size
        sum_ = mx.sum(self, axis=axis, keepdims=keepdims)
        if axis is None:
            shape = self.shape if hasattr(self, "shape") else mx.shape(self)
            from functools import reduce
            import operator
            size = reduce(operator.mul, shape, 1)
        else:
            # if axis is int or tuple, need to calculate reduced size
            if hasattr(self, "shape"):
                shape = self.shape
            else:
                shape = mx.shape(self)
            if isinstance(axis, int):
                axis_ = (axis,)
            else:
                axis_ = tuple(axis)
            import numpy as np
            size = 1
            for ax in axis_:
                size *= shape[ax]
        return sum_ / size

if not hasattr(mx.array, "mean"):
    mx.array.mean = _array_mean

# ----------- prod -----------
def _array_prod(self, axis=None, keepdims=False):
    """
    Return the product of array elements over the given axis.
    """
    if hasattr(mx, "prod"):
        return mx.prod(self, axis=axis, keepdims=keepdims)
    else:
        # fallback: repeated multiply
        import operator
        import functools
        if axis is None:
            flat = mx.reshape(self, (-1,))
            out = flat[0]
            for i in range(1, flat.shape[0]):
                out = out * flat[i]
            if keepdims:
                return mx.reshape(out, (1,) * len(self.shape))
            return out
        else:
            # fallback: convert tolist, use numpy, rewrap as mx.array
            import numpy as np
            arr = np.array(self.tolist())
            result = arr.prod(axis=axis, keepdims=keepdims)
            return mx.array(result)

if not hasattr(mx.array, "prod"):
    mx.array.prod = _array_prod

# ----------- min -----------
def _array_min(self, axis=None, keepdims=False):
    """
    Return the minimum of array elements over the given axis.
    """
    if hasattr(mx, "min"):
        return mx.min(self, axis=axis, keepdims=keepdims)
    else:
        # fallback: convert to numpy
        import numpy as np
        arr = np.array(self.tolist())
        result = arr.min(axis=axis, keepdims=keepdims)
        return mx.array(result)

if not hasattr(mx.array, "min"):
    mx.array.min = _array_min

# ----------- max -----------
def _array_max(self, axis=None, keepdims=False):
    """
    Return the maximum of array elements over the given axis.
    """
    if hasattr(mx, "max"):
        return mx.max(self, axis=axis, keepdims=keepdims)
    else:
        # fallback: convert to numpy
        import numpy as np
        arr = np.array(self.tolist())
        result = arr.max(axis=axis, keepdims=keepdims)
        return mx.array(result)

if not hasattr(mx.array, "max"):
    mx.array.max = _array_max

# ----------- all -----------
def _array_all(self, axis=None, keepdims=False):
    """
    Test whether all array elements along a given axis evaluate to True.
    """
    if hasattr(mx, "all"):
        return mx.all(self, axis=axis, keepdims=keepdims)
    else:
        import numpy as np
        arr = np.array(self.tolist())
        result = arr.all(axis=axis, keepdims=keepdims)
        return mx.array(result)

if not hasattr(mx.array, "all"):
    mx.array.all = _array_all

# ----------- any -----------
def _array_any(self, axis=None, keepdims=False):
    """
    Test whether any array element along a given axis evaluates to True.
    """
    if hasattr(mx, "any"):
        return mx.any(self, axis=axis, keepdims=keepdims)
    else:
        import numpy as np
        arr = np.array(self.tolist())
        result = arr.any(axis=axis, keepdims=keepdims)
        return mx.array(result)

if not hasattr(mx.array, "any"):
    mx.array.any = _array_any

# ----------- flatten -----------
def _array_flatten(self):
    """
    Return a copy of the array collapsed into one dimension.
    """
    size = 1
    shape = self.shape if hasattr(self, "shape") else mx.shape(self)
    for d in shape:
        size *= d
    return mx.reshape(self, (size,))

if not hasattr(mx.array, "flatten"):
    mx.array.flatten = _array_flatten

# ----------- reshape -----------
def _array_reshape(self, *shape):
    """
    Return a reshaped view of self.
    """
    # Allow reshape((3, 4)) or reshape(3, 4)
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        shape = tuple(shape[0])
    else:
        shape = tuple(shape)
    return mx.reshape(self, shape)

if not hasattr(mx.array, "reshape"):
    mx.array.reshape = _array_reshape

# ----------- unsqueeze -----------
def _array_unsqueeze(self, dim):
    """
    Return a new array with a dimension of size one inserted at the specified position.
    """
    return mx.expand_dims(self, axis=dim)

if not hasattr(mx.array, "unsqueeze"):
    mx.array.unsqueeze = _array_unsqueeze

# ----------- expand -----------
def _array_expand(self, *shape):
    """
    Return a new view of self broadcast to the given shape.
    """
    # Allow expand((3,4)) or expand(3,4)
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        shape = tuple(shape[0])
    else:
        shape = tuple(shape)
    return mx.broadcast_to(self, shape)

if not hasattr(mx.array, "expand"):
    mx.array.expand = _array_expand

# ----------- expand_as -----------
def _array_expand_as(self, other):
    """
    Expand self to the shape of another array.
    """
    shape = other.shape if hasattr(other, "shape") else mx.shape(other)
    return mx.broadcast_to(self, shape)

if not hasattr(mx.array, "expand_as"):
    mx.array.expand_as = _array_expand_as

# ----------- clone -----------
def _array_clone(self):
    """
    Return a copy of the array.
    """
    return mx.array(self)

if not hasattr(mx.array, "clone"):
    mx.array.clone = _array_clone

# ----------- detach -----------
def _array_detach(self):
    """
    Return a copy of the array (since gradients are not tracked in mlx).
    """
    return mx.array(self)

if not hasattr(mx.array, "detach"):
    mx.array.detach = _array_detach

# ----------- to (dtype only) -----------
def _array_to(self, *, dtype=None, **kwargs):
    """
    Return an array with the specified dtype (other arguments are ignored).
    """
    if dtype is None:
        return self
    return self.astype(dtype)

if not hasattr(mx.array, "to"):
    mx.array.to = _array_to

# ----------- numpy -----------
def _array_numpy(self):
    """
    Return a NumPy ndarray copy of the array. Raises informative error if numpy is not available.
    """
    try:
        import numpy as np
    except ImportError as e:
        raise RuntimeError("NumPy is required for .numpy(), but could not be imported: {}".format(e))
    return np.array(self.tolist())

if not hasattr(mx.array, "numpy"):
    mx.array.numpy = _array_numpy
