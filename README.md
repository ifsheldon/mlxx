# MLX eXtended

`mlx.core.array` supercharged.

## Usage

Installation:
* `pip install mlxx`
* `uv add mlxx`

After importing `mlx`, do this

```python
import mlx.core as mx
# this will monkey patch `mlx.core.array` class with more convenient methods
import mlxx as _ 
```

Then you can use some methods like `allclose`

```python
a = mx.array([1, 2, 3], dtype=mx.float32)
b = mx.array([1, 2, 3], dtype=mx.float32)
print(a.allclose(b))
print(a.inner(b))
c = mx.array([4,5,6])
print(a.add(c))
```

## Available Convenient Methods

### Comparison Operations
- `allclose(b, rtol=1e-05, atol=1e-08, equal_nan=False)`: Check if arrays are close within tolerance
- `isclose(b, rtol=1e-05, atol=1e-08, equal_nan=False)`: Element-wise comparison within tolerance
- `array_equal(b, equal_nan=False)`: Check if arrays are exactly equal

### Logical Operations
- `logical_and(b)`: Element-wise logical AND
- `logical_or(b)`: Element-wise logical OR
- `logical_not()`: Element-wise logical NOT

### Binary Operations
- `add(other)`: Element-wise addition
- `divide(other)` (alias: `div`): Element-wise division
- `multiply(other)` (alias: `mul`): Element-wise multiplication
- `logaddexp(other)`: Element-wise log(exp(a) + exp(b))
- `binary_maximum(b)`: Element-wise maximum of self and b.
- `binary_minimum(b)`: Element-wise minimum of self and b.
- `power(exponent)`: Element-wise power
- `matmul(b)`: Matrix multiplication
- `addmm(mat1, mat2, beta=1.0, alpha=1.0)`: Computes `beta * self + alpha * (mat1 @ mat2)`.
- `inner(b)`: Inner product

### Trigonometric Functions
- `arccos()`: Inverse cosine
- `arccosh()`: Inverse hyperbolic cosine
- `arcsin()`: Inverse sine
- `arcsinh()`: Inverse hyperbolic sine
- `arctan()`: Inverse tangent
- `arctanh()`: Inverse hyperbolic tangent
- `cosh()`: Hyperbolic cosine
- `sinh()`: Hyperbolic sine
- `tan()`: Tangent
- `tanh()`: Hyperbolic tangent

### Reduction Operations
- `nansum(axis=None, keepdims=False, dtype=None)`: Sum of array elements, treating NaNs as zero.

### Mathematical Functions
- `ceil()`: Ceiling function
- `floor()`: Floor function
- `degrees()`: Convert radians to degrees
- `radians()`: Convert degrees to radians
- `erf()`: Error function
- `erfinv()`: Inverse error function
- `expm1()`: exp(x) - 1
- `sigmoid()`: Sigmoid function
- `sign()`: Sign function
- `norm()`: Calculate norms, see `mlx.core.linalg.norm`

### Infinity and NaN Checks
- `isfinite()`: Check for finite values
- `isinf()`: Check for infinite values
- `isnan()`: Check for NaN values
- `isneginf()`: Check for negative infinity
- `isposinf()`: Check for positive infinity

### Other Operations
- `negative()`: Element-wise negation
- `stop_gradient()`: Stop gradient computation
- `permute(*axes)`: Permute axes (alias for transpose)
- `t()`: Transpose (alias for transpose)
- `numpy()`: Convert `mlx.core.array` to a `numpy.ndarray`

Note: All methods support an optional `stream` parameter for controlling computation streams where applicable.

## Contributing

Feel free to make PRs!

### Setup Dev Env
1. Run `uv sync`
2. Run `pre-commit install`, or `uv run pre-commit install` if your shell doesn't autodetect venv

## LICENSE
[MIT](LICENSE)