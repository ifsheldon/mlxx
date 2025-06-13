import mlx.core as mx
import numpy as np
import torch

# This import will apply the monkey patches
import mlxx as _


def assert_mx_equal_np(mx_arr, np_arr, msg=""):
    """Helper to compare MLX array with NumPy array."""
    assert mx_arr.dtype == mx.array(np_arr).dtype, (
        f"{msg} Dtype mismatch. MLX: {mx_arr.dtype}, NumPy expected: {mx.array(np_arr).dtype}"
    )
    np.testing.assert_array_equal(
        np.array(mx_arr), np_arr, err_msg=f"{msg} Array content mismatch"
    )


def assert_mx_allclose_np(mx_arr, np_arr, rtol=1e-05, atol=1e-08, msg=""):
    """Helper to compare MLX array with NumPy array using allclose."""
    assert mx_arr.dtype == mx.array(np_arr).dtype, (
        f"{msg} Dtype mismatch for allclose. MLX: {mx_arr.dtype}, NumPy expected: {mx.array(np_arr).dtype}"
    )
    np.testing.assert_allclose(
        np.array(mx_arr),
        np_arr,
        rtol=rtol,
        atol=atol,
        err_msg=f"{msg} Array content mismatch for allclose",
    )


def test_add():
    # Test case 1: Adding two MLX arrays
    a_mx = mx.array([1, 2, 3])
    b_mx = mx.array([4, 5, 6])
    expected_mx = mx.array([5, 7, 9])
    result_mx = a_mx.add(b_mx)
    assert mx.array_equal(result_mx, expected_mx), "Test Case 1 Failed: MLX arrays"

    # Test case 2: Adding MLX array and scalar
    a_mx_scalar = mx.array([1, 2, 3])
    b_scalar = 2
    expected_mx_scalar = mx.array([3, 4, 5])
    result_mx_scalar = a_mx_scalar.add(b_scalar)
    assert mx.array_equal(result_mx_scalar, expected_mx_scalar), (
        "Test Case 2 Failed: MLX array and scalar"
    )

    # Test case 3: Compare with PyTorch and NumPy
    a_np = np.array([1.0, 2.0, 3.0])
    b_np = np.array([4.0, 5.0, 6.0])
    expected_np = a_np + b_np
    a_torch = torch.tensor(a_np)
    b_torch = torch.tensor(b_np)
    expected_torch = a_torch.add(b_torch)

    a_mx_comp = mx.array(a_np)
    b_mx_comp = mx.array(b_np)
    result_mx_comp = a_mx_comp.add(b_mx_comp)

    assert_mx_equal_np(result_mx_comp, expected_np, "Test Case 3 Failed: MLX vs NumPy")
    assert_mx_equal_np(
        result_mx_comp, expected_torch.numpy(), "Test Case 3 Failed: MLX vs PyTorch"
    )


def test_addmm():
    # Test case 1: Basic addmm
    input_mx = mx.array([[1, 2], [3, 4]], dtype=mx.float32)
    mat1_mx = mx.array([[1, 2], [3, 4]], dtype=mx.float32)
    mat2_mx = mx.array([[5, 6], [7, 8]], dtype=mx.float32)
    expected_mx = mx.array([[20, 24], [46, 54]], dtype=mx.float32)
    result_mx = input_mx.addmm(mat1_mx, mat2_mx)
    assert mx.allclose(result_mx, expected_mx), "Test Case 1 Failed: Basic addmm"

    # Test case 2: addmm with beta and alpha
    expected_mx_beta_alpha = mx.array([[38.5, 45], [87.5, 102]], dtype=mx.float32)
    result_mx_beta_alpha = input_mx.addmm(mat1_mx, mat2_mx, beta=0.5, alpha=2.0)
    assert mx.allclose(result_mx_beta_alpha, expected_mx_beta_alpha), (
        "Test Case 2 Failed: addmm with beta and alpha"
    )

    # Test case 3: Compare with PyTorch and NumPy
    input_np = np.array([[1, 2], [3, 4]], dtype=np.float32)
    mat1_np = np.array([[1, 2], [3, 4]], dtype=np.float32)
    mat2_np = np.array([[5, 6], [7, 8]], dtype=np.float32)
    expected_np_addmm = input_np + (mat1_np @ mat2_np)

    input_torch = torch.tensor(input_np)
    mat1_torch = torch.tensor(mat1_np)
    mat2_torch = torch.tensor(mat2_np)
    expected_torch_addmm = input_torch.addmm(mat1_torch, mat2_torch)

    input_mx_comp = mx.array(input_np)
    mat1_mx_comp = mx.array(mat1_np)
    mat2_mx_comp = mx.array(mat2_np)
    result_mx_comp = input_mx_comp.addmm(mat1_mx_comp, mat2_mx_comp)

    assert_mx_allclose_np(
        result_mx_comp, expected_np_addmm, msg="Test Case 3 Failed: MLX vs NumPy addmm"
    )
    assert_mx_allclose_np(
        result_mx_comp,
        expected_torch_addmm.numpy(),
        msg="Test Case 3 Failed: MLX vs PyTorch addmm",
    )

    expected_np_beta_alpha = 0.5 * input_np + 2.0 * (mat1_np @ mat2_np)
    expected_torch_beta_alpha = input_torch.addmm(
        mat1_torch, mat2_torch, beta=0.5, alpha=2.0
    )
    result_mx_beta_alpha_comp = input_mx_comp.addmm(
        mat1_mx_comp, mat2_mx_comp, beta=0.5, alpha=2.0
    )
    assert_mx_allclose_np(
        result_mx_beta_alpha_comp,
        expected_np_beta_alpha,
        msg="Test Case 3 Failed: MLX vs NumPy addmm with beta/alpha",
    )
    assert_mx_allclose_np(
        result_mx_beta_alpha_comp,
        expected_torch_beta_alpha.numpy(),
        msg="Test Case 3 Failed: MLX vs PyTorch addmm with beta/alpha",
    )


def test_logaddexp():
    # Test case 1: Basic logaddexp
    a_mx = mx.array([1.0, 2.0], dtype=mx.float32)
    b_mx = mx.array([2.0, 1.0], dtype=mx.float32)
    expected_mx = mx.array([2.3132617, 2.3132617], dtype=mx.float32)
    result_mx = a_mx.logaddexp(b_mx)
    assert mx.allclose(result_mx, expected_mx, atol=1e-6), (
        "Test Case 1 Failed: Basic logaddexp"
    )

    # Test case 2: logaddexp with scalars
    a_mx_scalar = mx.array([1.0], dtype=mx.float32)
    b_scalar = 2.0
    expected_mx_scalar = mx.array([2.3132617], dtype=mx.float32)
    result_mx_scalar = a_mx_scalar.logaddexp(b_scalar)
    assert mx.allclose(result_mx_scalar, expected_mx_scalar, atol=1e-6), (
        "Test Case 2 Failed: logaddexp with scalar"
    )

    # Test case 3: Compare with PyTorch and NumPy
    a_np = np.array([1.0, 2.0, -1.0], dtype=np.float32)
    b_np = np.array([2.0, 1.0, 0.5], dtype=np.float32)
    expected_np_logaddexp = np.logaddexp(a_np, b_np)

    a_torch = torch.tensor(a_np)
    b_torch = torch.tensor(b_np)
    expected_torch_logaddexp = torch.logaddexp(a_torch, b_torch)

    a_mx_comp = mx.array(a_np)
    b_mx_comp = mx.array(b_np)
    result_mx_comp = a_mx_comp.logaddexp(b_mx_comp)

    assert_mx_allclose_np(
        result_mx_comp,
        expected_np_logaddexp,
        msg="Test Case 3 Failed: MLX vs NumPy logaddexp",
        atol=1e-5,
    )
    assert_mx_allclose_np(
        result_mx_comp,
        expected_torch_logaddexp.numpy(),
        msg="Test Case 3 Failed: MLX vs PyTorch logaddexp",
        atol=1e-5,
    )


def test_multiply():
    # Test case 1: Multiplying two MLX arrays
    a_mx = mx.array([1, 2, 3])
    b_mx = mx.array([4, 5, 6])
    expected_mx = mx.array([4, 10, 18])
    result_mx = a_mx.multiply(b_mx)
    assert mx.array_equal(result_mx, expected_mx), (
        "Test Case 1 Failed: MLX arrays multiply"
    )
    result_mx_mul = a_mx.mul(b_mx)
    assert mx.array_equal(result_mx_mul, expected_mx), (
        "Test Case 1 Failed: MLX arrays mul alias"
    )

    # Test case 2: Multiplying MLX array and scalar
    a_mx_scalar = mx.array([1, 2, 3])
    b_scalar = 2
    expected_mx_scalar = mx.array([2, 4, 6])
    result_mx_scalar = a_mx_scalar.multiply(b_scalar)
    assert mx.array_equal(result_mx_scalar, expected_mx_scalar), (
        "Test Case 2 Failed: MLX array and scalar multiply"
    )

    # Test case 3: Compare with PyTorch and NumPy
    a_np = np.array([1.0, 2.0, 3.0])
    b_np = np.array([4.0, 5.0, 6.0])
    expected_np = a_np * b_np
    a_torch = torch.tensor(a_np)
    b_torch = torch.tensor(b_np)
    expected_torch = a_torch.multiply(b_torch)

    a_mx_comp = mx.array(a_np)
    b_mx_comp = mx.array(b_np)
    result_mx_comp = a_mx_comp.multiply(b_mx_comp)

    assert_mx_equal_np(
        result_mx_comp, expected_np, "Test Case 3 Failed: MLX vs NumPy multiply"
    )
    assert_mx_equal_np(
        result_mx_comp,
        expected_torch.numpy(),
        "Test Case 3 Failed: MLX vs PyTorch multiply",
    )


def test_nansum():
    # Test case 1: nansum with NaNs
    a_mx = mx.array([1.0, mx.nan, 3.0], dtype=mx.float32)
    expected_mx = mx.array(4.0, dtype=mx.float32)
    result_mx = a_mx.nansum()
    assert mx.array_equal(result_mx, expected_mx), (
        "Test Case 1 Failed: nansum with NaNs"
    )

    # Test case 2: nansum along an axis
    a_mx_axis = mx.array([[1.0, mx.nan], [3.0, 4.0]], dtype=mx.float32)
    expected_mx_axis0 = mx.array([4.0, 4.0], dtype=mx.float32)
    result_mx_axis0 = a_mx_axis.nansum(axis=0)
    assert mx.array_equal(result_mx_axis0, expected_mx_axis0), (
        "Test Case 2 Failed: nansum axis 0"
    )

    expected_mx_axis1 = mx.array([1.0, 7.0], dtype=mx.float32)
    result_mx_axis1 = a_mx_axis.nansum(axis=1)
    assert mx.array_equal(result_mx_axis1, expected_mx_axis1), (
        "Test Case 2 Failed: nansum axis 1"
    )

    # Test case 3: Compare with PyTorch and NumPy
    a_np = np.array([[1.0, np.nan, 2.0], [np.nan, 3.0, np.nan]], dtype=np.float32)
    expected_np_nansum_flat = np.nansum(a_np)
    expected_np_nansum_axis0 = np.nansum(a_np, axis=0)
    expected_np_nansum_axis1 = np.nansum(a_np, axis=1)

    a_torch = torch.tensor(a_np)
    expected_torch_nansum_flat = torch.nansum(a_torch)
    expected_torch_nansum_axis0 = torch.nansum(a_torch, dim=0)
    expected_torch_nansum_axis1 = torch.nansum(a_torch, dim=1)

    a_mx_comp = mx.array(a_np)
    result_mx_comp_flat = a_mx_comp.nansum()
    result_mx_comp_axis0 = a_mx_comp.nansum(axis=0)
    result_mx_comp_axis1 = a_mx_comp.nansum(axis=1)

    assert_mx_allclose_np(
        result_mx_comp_flat,
        expected_np_nansum_flat,
        msg="Test Case 3 Failed: MLX vs NumPy nansum flat",
    )
    if expected_torch_nansum_flat.dtype != torch.float32:
        expected_torch_nansum_flat = expected_torch_nansum_flat.to(torch.float32)
    assert_mx_allclose_np(
        result_mx_comp_flat,
        expected_torch_nansum_flat.numpy(),
        msg="Test Case 3 Failed: MLX vs PyTorch nansum flat",
    )

    assert_mx_allclose_np(
        result_mx_comp_axis0,
        expected_np_nansum_axis0,
        msg="Test Case 3 Failed: MLX vs NumPy nansum axis0",
    )
    if expected_torch_nansum_axis0.dtype != torch.float32:
        expected_torch_nansum_axis0 = expected_torch_nansum_axis0.to(torch.float32)
    assert_mx_allclose_np(
        result_mx_comp_axis0,
        expected_torch_nansum_axis0.numpy(),
        msg="Test Case 3 Failed: MLX vs PyTorch nansum axis0",
    )

    assert_mx_allclose_np(
        result_mx_comp_axis1,
        expected_np_nansum_axis1,
        msg="Test Case 3 Failed: MLX vs NumPy nansum axis1",
    )
    if expected_torch_nansum_axis1.dtype != torch.float32:
        expected_torch_nansum_axis1 = expected_torch_nansum_axis1.to(torch.float32)
    assert_mx_allclose_np(
        result_mx_comp_axis1,
        expected_torch_nansum_axis1.numpy(),
        msg="Test Case 3 Failed: MLX vs PyTorch nansum axis1",
    )


def test_divide():
    # Test case 1: Dividing two MLX arrays (float)
    a_mx_f = mx.array([4.0, 10.0, 18.0], dtype=mx.float32)
    b_mx_f = mx.array([2.0, 5.0, 3.0], dtype=mx.float32)
    expected_mx_f = mx.array([2.0, 2.0, 6.0], dtype=mx.float32)
    result_mx_f = a_mx_f.divide(b_mx_f)
    assert mx.allclose(result_mx_f, expected_mx_f), (
        "Test Case 1 Failed: MLX float arrays divide"
    )
    result_mx_f_div = a_mx_f.div(b_mx_f)
    assert mx.allclose(result_mx_f_div, expected_mx_f), (
        "Test Case 1 Failed: MLX float arrays div alias"
    )

    # Test case 2: Dividing MLX array and scalar (float)
    a_mx_scalar_f = mx.array([2.0, 4.0, 6.0], dtype=mx.float32)
    b_scalar_f = 2.0
    expected_mx_scalar_f = mx.array([1.0, 2.0, 3.0], dtype=mx.float32)
    result_mx_scalar_f = a_mx_scalar_f.divide(b_scalar_f)
    assert mx.allclose(result_mx_scalar_f, expected_mx_scalar_f), (
        "Test Case 2 Failed: MLX float array and scalar divide"
    )

    # Test case 3: Compare with PyTorch and NumPy (float)
    a_np_f = np.array([4.0, 10.0, 18.0], dtype=np.float32)
    b_np_f = np.array([2.0, 5.0, 3.0], dtype=np.float32)
    expected_np_f = a_np_f / b_np_f
    a_torch_f = torch.tensor(a_np_f)
    b_torch_f = torch.tensor(b_np_f)
    expected_torch_f = a_torch_f.divide(b_torch_f)

    a_mx_comp_f = mx.array(a_np_f)
    b_mx_comp_f = mx.array(b_np_f)
    result_mx_comp_f = a_mx_comp_f.divide(b_mx_comp_f)

    assert_mx_allclose_np(
        result_mx_comp_f,
        expected_np_f,
        msg="Test Case 3 Failed: MLX vs NumPy float divide",
    )
    assert_mx_allclose_np(
        result_mx_comp_f,
        expected_torch_f.numpy(),
        msg="Test Case 3 Failed: MLX vs PyTorch float divide",
    )

    # Test case 4: Integer division
    a_mx_i = mx.array([4, 10, 18], dtype=mx.int32)
    b_mx_i = mx.array([2, 5, 3], dtype=mx.int32)
    expected_mx_i = mx.array([2, 2, 6], dtype=mx.int32)
    result_mx_i = a_mx_i.divide(b_mx_i)
    assert mx.array_equal(result_mx_i, expected_mx_i), (
        "Test Case 4 Failed: MLX integer arrays exact divide"
    )


def test_allclose():
    a = mx.array([1.0, 2.0, 3.0])
    b = mx.array([1.00000001, 2.00000001, 3.00000001])
    assert a.allclose(b)  # default tolerances
    c = mx.array([1.1, 2.1, 3.1])
    assert not a.allclose(c)
    d = mx.array([1.0, 2.0, mx.nan])
    e = mx.array([1.0, 2.0, mx.nan])
    assert d.allclose(e, equal_nan=True)
    assert not d.allclose(e, equal_nan=False)


def test_isclose():
    a = mx.array([1.0, 2.0, 3.0])
    b = mx.array([1.00000001, 2.1, 3.0])
    expected = mx.array([True, False, True])
    assert mx.array_equal(a.isclose(b), expected)
    c = mx.array([1.0, 2.0, mx.nan])
    d = mx.array([1.0, 2.0, mx.nan])
    assert mx.array_equal(c.isclose(d, equal_nan=True), mx.array([True, True, True]))
    assert mx.array_equal(c.isclose(d, equal_nan=False), mx.array([True, True, False]))


def test_array_equal():
    a = mx.array([1, 2, 3])
    b = mx.array([1, 2, 3])
    assert a.array_equal(b)
    c = mx.array([1, 2, 4])
    assert not a.array_equal(c)
    d = mx.array([1, 2, mx.nan])
    e = mx.array([1, 2, mx.nan])
    assert d.array_equal(e, equal_nan=True)
    assert not d.array_equal(e, equal_nan=False)


def test_logical_and():
    a = mx.array([True, True, False, False])
    b = mx.array([True, False, True, False])
    expected = mx.array([True, False, False, False])
    assert mx.array_equal(a.logical_and(b), expected)


def test_logical_or():
    a = mx.array([True, True, False, False])
    b = mx.array([True, False, True, False])
    expected = mx.array([True, True, True, False])
    assert mx.array_equal(a.logical_or(b), expected)


def test_logical_not():
    a = mx.array([True, False])
    expected = mx.array([False, True])
    assert mx.array_equal(a.logical_not(), expected)


def test_binary_maximum():
    a = mx.array([1, 5, 3])
    b = mx.array([4, 2, 6])
    expected = mx.array([4, 5, 6])
    assert mx.array_equal(a.binary_maximum(b), expected)
    assert mx.array_equal(mx.maximum(a, b), expected)  # Also test mx.maximum directly


def test_binary_minimum():
    a = mx.array([1, 5, 3])
    b = mx.array([4, 2, 6])
    expected = mx.array([1, 2, 3])
    assert mx.array_equal(a.binary_minimum(b), expected)
    assert mx.array_equal(mx.minimum(a, b), expected)  # Also test mx.minimum directly


def test_power():
    a = mx.array([1, 2, 3], dtype=mx.float32)
    exponent = mx.array([2, 3, 2], dtype=mx.float32)
    expected = mx.array([1, 8, 9], dtype=mx.float32)
    assert mx.allclose(a.power(exponent), expected)
    assert mx.allclose(a.power(2), mx.array([1, 4, 9], dtype=mx.float32))


def test_matmul():
    a = mx.array([[1, 2], [3, 4]], dtype=mx.float32)
    b = mx.array([[5, 6], [7, 8]], dtype=mx.float32)
    expected = mx.array([[19, 22], [43, 50]], dtype=mx.float32)
    assert mx.array_equal(a.matmul(b), expected)


def test_inner():
    a = mx.array([1, 2, 3], dtype=mx.float32)
    b = mx.array([4, 5, 6], dtype=mx.float32)
    # 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
    expected = mx.array(32, dtype=mx.float32)
    assert mx.array_equal(a.inner(b), expected)

    x = mx.array([[1, 2], [3, 4]], dtype=mx.float32)
    y = mx.array([[5, 6], [7, 8]], dtype=mx.float32)
    # [[1,2] . [5,6], [1,2] . [7,8]] = [[5+12, 7+16]] = [[17, 23]]
    # [[3,4] . [5,6], [3,4] . [7,8]] = [[15+24, 21+32]] = [[39, 53]]
    expected_2d = mx.array([[17, 23], [39, 53]], dtype=mx.float32)
    assert mx.array_equal(x.inner(y), expected_2d)


def test_arc_trig_hyperbolic():
    # Test values chosen to be within valid domains and give somewhat clean results where possible
    # mx.arccos
    a_cos = mx.array([0, 1, -1], dtype=mx.float32)
    # PyTorch: tensor([1.5708, 0.0000, 3.1416])
    # NumPy:   array([1.5707964, 0.       , 3.1415927], dtype=float32)
    expected_arccos = mx.array([np.pi / 2, 0, np.pi], dtype=mx.float32)
    assert mx.allclose(a_cos.arccos(), expected_arccos, atol=1e-6)

    # mx.arccosh
    a_cosh = mx.array([1, 2, 3], dtype=mx.float32)
    # PyTorch: tensor([0.0000, 1.3170, 1.7627])
    # NumPy:   array([0.       , 1.316958 , 1.7627472], dtype=float32)
    expected_arccosh = mx.array(
        np.arccosh(np.array([1, 2, 3], dtype=np.float32)), dtype=mx.float32
    )
    assert mx.allclose(a_cosh.arccosh(), expected_arccosh, atol=1e-6)

    # mx.arcsin
    a_sin = mx.array([0, 1, -1], dtype=mx.float32)
    expected_arcsin = mx.array([0, np.pi / 2, -np.pi / 2], dtype=mx.float32)
    assert mx.allclose(a_sin.arcsin(), expected_arcsin, atol=1e-6)

    # mx.arcsinh
    a_sinh = mx.array([0, 1, -1], dtype=mx.float32)
    expected_arcsinh = mx.array(
        np.arcsinh(np.array([0, 1, -1], dtype=np.float32)), dtype=mx.float32
    )
    assert mx.allclose(a_sinh.arcsinh(), expected_arcsinh, atol=1e-6)

    # mx.arctan
    a_tan = mx.array([0, 1, -1], dtype=mx.float32)
    expected_arctan = mx.array([0, np.pi / 4, -np.pi / 4], dtype=mx.float32)
    assert mx.allclose(a_tan.arctan(), expected_arctan, atol=1e-6)

    # mx.arctanh
    a_tanh = mx.array([0, 0.5, -0.5], dtype=mx.float32)
    expected_arctanh = mx.array(
        np.arctanh(np.array([0, 0.5, -0.5], dtype=np.float32)), dtype=mx.float32
    )
    assert mx.allclose(a_tanh.arctanh(), expected_arctanh, atol=1e-6)


def test_ceil_floor():
    a = mx.array([1.2, 2.0, 3.7, -0.5, -2.0], dtype=mx.float32)

    expected_ceil = mx.array([2.0, 2.0, 4.0, 0.0, -2.0], dtype=mx.float32)
    assert mx.array_equal(a.ceil(), expected_ceil)

    expected_floor = mx.array([1.0, 2.0, 3.0, -1.0, -2.0], dtype=mx.float32)
    assert mx.array_equal(a.floor(), expected_floor)


def test_hyperbolic_trig():
    a = mx.array([0, 1, -1], dtype=mx.float32)
    a_np = np.array([0, 1, -1], dtype=np.float32)

    # cosh
    expected_cosh = mx.array(np.cosh(a_np), dtype=mx.float32)
    assert mx.allclose(a.cosh(), expected_cosh, atol=1e-6)

    # sinh
    expected_sinh = mx.array(np.sinh(a_np), dtype=mx.float32)
    assert mx.allclose(a.sinh(), expected_sinh, atol=1e-6)

    # tanh
    expected_tanh = mx.array(np.tanh(a_np), dtype=mx.float32)
    assert mx.allclose(a.tanh(), expected_tanh, atol=1e-6)

    # tan
    b = mx.array([0, np.pi / 4, -np.pi / 4], dtype=mx.float32)
    b_np = np.array([0, np.pi / 4, -np.pi / 4], dtype=np.float32)
    expected_tan = mx.array(np.tan(b_np), dtype=mx.float32)
    assert mx.allclose(b.tan(), expected_tan, atol=1e-6)


def test_degrees_radians():
    # radians to degrees
    rad = mx.array([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi], dtype=mx.float32)
    expected_deg = mx.array([0, 90, 180, 270, 360], dtype=mx.float32)
    assert mx.allclose(
        rad.degrees(), expected_deg, atol=1e-4
    )  # Increased atol for pi inaccuracies

    # degrees to radians
    deg = mx.array([0, 90, 180, 270, 360], dtype=mx.float32)
    expected_rad = mx.array(
        [0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi], dtype=mx.float32
    )
    assert mx.allclose(deg.radians(), expected_rad, atol=1e-4)


def test_erf_erfinv_expm1():
    # Test values for erf
    erf_inputs_np = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
    erf_inputs_mx = mx.array(erf_inputs_np)
    # Expected values for erf from numpy/scipy: scipy.special.erf([-2., -1., 0., 1., 2.])
    # array([-0.9953222 , -0.84270075,  0.        ,  0.84270075,  0.9953222 ], dtype=float32)
    expected_erf_np = np.array(
        [-0.9953222, -0.84270075, 0.0, 0.84270075, 0.9953222], dtype=np.float32
    )
    assert_mx_allclose_np(
        erf_inputs_mx.erf(), expected_erf_np, msg="erf check failed", atol=1e-6
    )

    # Test values for erfinv - chosen such that erf(erfinv(x)) = x
    # Input values are outputs of erf, so erfinv should return the original inputs to erf (approx)
    erfinv_inputs_np = np.array(
        [-0.84270075, 0.0, 0.84270075], dtype=np.float32
    )  # erf(-1), erf(0), erf(1)
    erfinv_inputs_mx = mx.array(erfinv_inputs_np)
    expected_erfinv_np = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
    assert_mx_allclose_np(
        erfinv_inputs_mx.erfinv(),
        expected_erfinv_np,
        msg="erfinv check failed",
        atol=1e-5,
    )

    # expm1
    expm1_inputs_np = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
    expm1_inputs_mx = mx.array(expm1_inputs_np)
    expected_expm1_np = np.expm1(expm1_inputs_np)
    assert_mx_allclose_np(
        expm1_inputs_mx.expm1(), expected_expm1_np, msg="expm1 check failed", atol=1e-6
    )


def test_is_conditions():
    a_np = np.array([1.0, np.nan, np.inf, -np.inf, 0.0], dtype=np.float32)
    a_mx = mx.array(a_np)

    assert mx.array_equal(a_mx.isfinite(), mx.array([True, False, False, False, True]))
    assert mx.array_equal(a_mx.isinf(), mx.array([False, False, True, True, False]))
    assert mx.array_equal(a_mx.isnan(), mx.array([False, True, False, False, False]))
    # MLX does not have specific isneginf/isposinf functions; mx.isinf covers both.
    # We can test the patched versions by checking against (mx.isinf(x) & (x < 0)) and (mx.isinf(x) & (x > 0))
    # However, the current patches directly call mx.isneginf and mx.isposinf which might not exist.
    # Let's verify what mx.isneginf and mx.isposinf actually do in the version of MLX being used.
    # If they are not present, these tests might fail or the patch needs adjustment.

    # Assuming mx.isneginf and mx.isposinf are indeed patched to work as expected:
    expected_isneginf = mx.array([False, False, False, True, False])
    assert mx.array_equal(a_mx.isneginf(), expected_isneginf), "isneginf check failed"

    expected_isposinf = mx.array([False, False, True, False, False])
    assert mx.array_equal(a_mx.isposinf(), expected_isposinf), "isposinf check failed"


def test_negative_sign_sigmoid():
    a_np = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
    a_mx = mx.array(a_np)

    # negative
    expected_negative_np = np.negative(a_np)
    assert_mx_allclose_np(
        a_mx.negative(), expected_negative_np, msg="negative check failed"
    )

    # sign
    expected_sign_np = np.sign(a_np)
    assert_mx_allclose_np(a_mx.sign(), expected_sign_np, msg="sign check failed")

    # sigmoid
    # sigmoid(x) = 1 / (1 + exp(-x))
    expected_sigmoid_np = 1 / (1 + np.exp(-a_np))
    assert_mx_allclose_np(
        a_mx.sigmoid(), expected_sigmoid_np, msg="sigmoid check failed", atol=1e-6
    )


def test_stop_gradient():
    # This is hard to test without autograd context, but we can ensure it returns an identical array.
    a = mx.array([1.0, 2.0, 3.0])
    b = a.stop_gradient()
    assert mx.array_equal(a, b)
    assert a.dtype == b.dtype
    # Further testing would require a gradient computation framework.


def test_permute():
    a = mx.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])  # shape (2,2,2)
    # Equivalent to transpose(0,2,1)
    expected_pytorch_permute = torch.tensor(np.array(a)).permute(0, 2, 1).numpy()

    # Test with explicit axes
    permuted_a = a.permute(0, 2, 1)
    assert_mx_equal_np(permuted_a, expected_pytorch_permute, "permute(0,2,1) failed")
    assert permuted_a.shape == (2, 2, 2)

    # Test default permute (transpose)
    b = mx.array([[1, 2, 3], [4, 5, 6]])  # shape (2,3)
    expected_transpose = np.array(b).T
    permuted_b_default = b.permute()
    assert_mx_equal_np(
        permuted_b_default, expected_transpose, "default permute (transpose) failed"
    )
    assert permuted_b_default.shape == (3, 2)

    a = mx.random.normal((2, 3))
    a = a.t()
    assert a.shape == (3, 2)


def test_norm():
    a = mx.random.normal((2, 3))
    assert mx.array_equal(a.norm(), mx.linalg.norm(a))
    a = mx.random.normal((2, 3), dtype=mx.complex64)
    assert mx.array_equal(a.norm(), mx.linalg.norm(a))
