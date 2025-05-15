# MLX eXtended

`mlx.core.array` supercharged.

## Usage

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
```

## Contributing

Feel free to make PRs!

### Setup Dev Env
1. Run `uv sync`
2. Run `pre-commit install`, or `uv run pre-commit install` if your shell doesn't autodetect venv

## LICENSE
[MIT](LICENSE)