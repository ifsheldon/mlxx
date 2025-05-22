import mlx.core as mx

# PyTorch/NumPy-style monkey-patches for non-native methods

def _array_permute(self, *dims):
    """
    Returns an array with axes permuted in the given order.
    Equivalent to numpy.transpose with the axes argument.

    Args:
        *dims: The desired ordering of dimensions.

    Returns:
        Permuted array.
    """
    if len(dims) == 1 and isinstance(dims[0], (tuple, list)):
        dims = dims[0]
    return mx.transpose(self, axes=dims)

if not hasattr(mx.array, "permute"):
    mx.array.permute = _array_permute

def _array_clone(self):
    """
    Returns a deep copy of the array.
    """
    return mx.array(self)

if not hasattr(mx.array, "clone"):
    mx.array.clone = _array_clone

def _array_fill_(self, value):
    """
    Returns a new array with the same shape as self, filled with the scalar value.
    Note: In-place operation is not supported in MLX, so this returns a new array.
    """
    return mx.full_like(self, value)

if not hasattr(mx.array, "fill_"):
    mx.array.fill_ = _array_fill_

def _array_unique(self):
    """
    Returns the unique values in the array, sorted. Falls back to NumPy if mx.unique is unavailable.
    """
    try:
        return mx.unique(self)
    except AttributeError:
        import numpy as np
        return mx.array(np.unique(self.__array__()))

if not hasattr(mx.array, "unique"):
    mx.array.unique = _array_unique

def _array_index_select(self, axis, index):
    """
    Selects elements along a given axis using an index array (like numpy.take).

    Args:
        axis: Axis along which to select.
        index: Indices of elements to select (1-D array or list).

    Returns:
        Selected array.
    """
    return mx.take(self, index, axis=axis)

if not hasattr(mx.array, "index_select"):
    mx.array.index_select = _array_index_select

def _array_masked_select(self, mask):
    """
    Returns elements where a boolean mask is True.

    Args:
        mask: Boolean array of the same shape as self.

    Returns:
        1D array of selected elements.
    """
    try:
        # Try direct boolean indexing
        return self[mask]
    except Exception:
        # Fallback to NumPy workaround if direct indexing fails
        import numpy as np
        arr = self.__array__()
        mask_np = mask.__array__() if hasattr(mask, '__array__') else np.array(mask)
        return mx.array(arr[mask_np])

if not hasattr(mx.array, "masked_select"):
    mx.array.masked_select = _array_masked_select

def _array_gather(self, axis, index):
    """
    Gathers values along a given axis according to indices (like numpy.take_along_axis).

    Args:
        axis: Axis along which to gather.
        index: Indices to gather, must be broadcastable to self.shape.

    Returns:
        Gathered array.
    """
    return mx.take_along_axis(self, index, axis=axis)

if not hasattr(mx.array, "gather"):
    mx.array.gather = _array_gather