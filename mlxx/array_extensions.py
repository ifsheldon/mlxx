import mlx.core as mx

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

# ----------- permute (alias for transpose) -----------
if not hasattr(mx.array, "permute"):
    mx.array.permute = mx.array.transpose