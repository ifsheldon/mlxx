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