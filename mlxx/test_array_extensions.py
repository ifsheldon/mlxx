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