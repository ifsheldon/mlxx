import mlx.core as mx
import mlxx as _

if __name__ == "__main__":
    a = mx.array([1, 2, 3], dtype=mx.float32)
    b = mx.array([1, 2, 3], dtype=mx.float32)
    print(a.allclose(b))
    print(a.inner(b))
