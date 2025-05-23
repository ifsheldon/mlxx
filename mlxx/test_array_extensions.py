import pytest
import mlx.core as mx
import numpy as np
import mlxx.array_extensions  # ensure monkey patching is applied

@pytest.mark.parametrize("data", [
    [1, 2, 3],
    [[4, 5], [6, 7]],
    [[[1], [2]], [[3], [4]]],
    42,
    [[1.5, 2.5], [3.5, 4.5]],
])
def test_mlx_array_tolist_matches_numpy(data):
    mlx_arr = mx.array(data)
    np_arr = np.array(data)
    assert mlx_arr.tolist() == np_arr.tolist()

@pytest.mark.parametrize("data", [
    [-1, 0, 2],
    [[-3, 4], [5, -6]],
    [[[-1], [2]], [[-3], [4]]],
    -42,
    [[-1.5, 2.5], [3.5, -4.5]],
])
def test_mlx_array_abs_matches_numpy_and_mxabs(data):
    mlx_arr = mx.array(data)
    np_arr = np.array(data)
    # Compare to numpy
    assert mlx_arr.abs().tolist() == np.abs(np_arr).tolist()
    # Compare to mx.abs function directly
    assert mlx_arr.abs().tolist() == mx.abs(mlx_arr).tolist()

@pytest.mark.parametrize("data", [
    [1, 2, 3],
    [[2, 3], [4, 5]],
    [[[-1], [2]], [[-3], [4]]],
    [0.5, 2.0, 4.0],
])
def test_mlx_array_prod_matches_numpy_and_mxprod(data):
    mlx_arr = mx.array(data)
    np_arr = np.array(data)
    # Default: full product
    assert mlx_arr.prod().item() == np.prod(np_arr).item() == mx.prod(mlx_arr).item()
    # Test axis=0 if possible
    if np_arr.ndim > 0:
        mlx_axis0 = mlx_arr.prod(axis=0)
        np_axis0 = np.prod(np_arr, axis=0)
        mx_axis0 = mx.prod(mlx_arr, axis=0)
        assert mlx_axis0.tolist() == np_axis0.tolist() == mx_axis0.tolist()
    # Test axis=-1 if possible and >1D
    if np_arr.ndim > 1:
        mlx_axis1 = mlx_arr.prod(axis=-1)
        np_axis1 = np.prod(np_arr, axis=-1)
        mx_axis1 = mx.prod(mlx_arr, axis=-1)
        assert mlx_axis1.tolist() == np_axis1.tolist() == mx_axis1.tolist()

@pytest.mark.parametrize("data", [
    [0, 0, 0],
    [0, 1, 0],
    [[0, 0], [0, 1]],
    [[True, False], [False, False]],
    [[0, 0], [0, 0]],
    [[], []],
    [[0.0, 0.0], [0.0, 0.0]],
])
def test_mlx_array_any_matches_numpy_and_mxany(data):
    mlx_arr = mx.array(data)
    np_arr = np.array(data)
    # Default: full any (scalar)
    assert bool(mlx_arr.any().item()) == bool(np.any(np_arr)) == bool(mx.any(mlx_arr).item())
    # Test axis=0 if possible
    if np_arr.ndim > 0 and np_arr.size > 0:
        mlx_axis0 = mlx_arr.any(axis=0)
        np_axis0 = np.any(np_arr, axis=0)
        mx_axis0 = mx.any(mlx_arr, axis=0)
        assert mlx_axis0.tolist() == np_axis0.tolist() == mx_axis0.tolist()
    # Test axis=-1 if possible and >1D
    if np_arr.ndim > 1 and np_arr.size > 0:
        mlx_axis1 = mlx_arr.any(axis=-1)
        np_axis1 = np.any(np_arr, axis=-1)
        mx_axis1 = mx.any(mlx_arr, axis=-1)
        assert mlx_axis1.tolist() == np_axis1.tolist() == mx_axis1.tolist()

@pytest.mark.parametrize("data", [
    [1, 2, 3],
    [[2, 3], [4, 5]],
    [[-1, 0], [0, 1]],
    [0.5, 2.0, 4.0],
])
def test_mlx_array_clone_value_equality_and_independence(data):
    mlx_arr = mx.array(data)
    clone = mlx_arr.clone()
    # Value equality
    assert mx.array_equal(mlx_arr, clone)
    # Not the same object
    assert clone is not mlx_arr
    # Changing clone does not change original (where possible)
    if mlx_arr.size > 0:
        # Add 1 to all elements in clone, original should not change
        clone_plus = clone + 1
        assert not mx.array_equal(mlx_arr, clone_plus)
        assert mx.array_equal(mlx_arr, mx.array(data))  # Original unchanged

@pytest.mark.parametrize(
    "data, new_shape",
    [
        ([1, 2, 3, 4], (2, 2)),
        ([[1, 2], [3, 4]], (4,)),
        ([[[1], [2]], [[3], [4]]], (2, 2)),
        ([0.0, 1.0, 2.0, 3.0], (2, 2)),
    ],
)
def test_mlx_array_reshape_matches_numpy_and_mxreshape(data, new_shape):
    mlx_arr = mx.array(data)
    np_arr = np.array(data)
    # MLX monkey-patched
    mlx_reshaped = mlx_arr.reshape(new_shape)
    # MLX functional
    mx_reshaped = mx.reshape(mlx_arr, new_shape)
    # NumPy
    np_reshaped = np_arr.reshape(new_shape)
    assert mlx_reshaped.tolist() == np_reshaped.tolist() == mx_reshaped.tolist()
    assert mlx_reshaped.shape == np_reshaped.shape == mx_reshaped.shape

@pytest.mark.parametrize(
    "data, new_shape",
    [
        ([1, 2, 3, 4], (2, 2)),
        ([[1, 2], [3, 4]], (4,)),
        ([0, 1, 2, 3, 4, 5], (3, 2)),
        ([7.0, 8.0, 9.0, 10.0], (2, 2)),
    ]
)
def test_mlx_array_view_matches_reshape(data, new_shape):
    mlx_arr = mx.array(data)
    reshaped = mlx_arr.reshape(new_shape)
    viewed = mlx_arr.view(new_shape)
    assert viewed.tolist() == reshaped.tolist()
    assert viewed.shape == reshaped.shape

@pytest.mark.parametrize(
    "data, axis",
    [
        ([1, 2, 3], 0),
        ([1, 2, 3], 1),
        ([[1, 2], [3, 4]], 0),
        ([[1, 2], [3, 4]], 1),
        ([[1, 2], [3, 4]], 2),
    ]
)
def test_mlx_array_unsqueeze_matches_numpy_and_mxexpand_dims(data, axis):
    mlx_arr = mx.array(data)
    np_arr = np.array(data)
    # MLX monkey-patched
    unsq = mlx_arr.unsqueeze(axis)
    # MLX expand_dims
    mx_unsq = mx.expand_dims(mlx_arr, axis)
    # NumPy expand_dims
    np_unsq = np.expand_dims(np_arr, axis)
    assert unsq.tolist() == np_unsq.tolist() == mx_unsq.tolist()
    assert unsq.shape == np_unsq.shape == mx_unsq.shape