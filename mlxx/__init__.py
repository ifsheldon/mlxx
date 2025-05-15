"""
mlxx: MLX Extensions

This package extends the _mx.array class with additional methods
that wrap binary operations from _mx. This allows for a more
object-oriented style of programming with MLX arrays.

To use, simply import mlxx, and the new methods will be available
on any _mx.array instance:

    import mlx.core as mx
    import mlxx  # This import applies the extensions

    a = mx.array([1, 2, 3])
    b = mx.array([1, 2, 4])

    # Now you can use the extended methods:
    print(a.allclose(b))
    print(a.logical_and(mx.array([True, True, False])))

"""

__all__ = []  # Explicitly declare that `from mlxx import *` imports nothing directly.

from .array_extensions import *  # This will execute the patching code within array_extensions.py
