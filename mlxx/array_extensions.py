import mlx.core as mx

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