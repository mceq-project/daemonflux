import numpy as np
import numpy.testing as npt
from daemonflux.flux import Parameters, _FluxEntry
from daemonflux.utils import format_angle, rearrange_covariance, is_iterable, grid_cov


def test_format_angle():
    tests = [
        (0.12345, "0.1235"),
        (1.23456, "1.2346"),
        (12.34567, "12.3457"),
        (-0.12345, "-0.1235"),
        (-1.23456, "-1.2346"),
        (-12.34567, "-12.3457"),
    ]
    for ang, expected in tests:
        result = format_angle(ang)
        assert (
            result == expected
        ), f"For angle {ang}, expected {expected} but got {result}"


def test_parameters_invcov():
    known_parameters = ["p1", "p2"]
    values = np.array([1, 2])
    cov = np.array([[1.0, 0.5], [0.5, 1.0]])
    params = Parameters(known_parameters, values, cov)
    expected_invcov = np.array([[1.333333, -0.666667], [-0.666667, 1.333333]])
    npt.assert_allclose(params.invcov, expected_invcov, rtol=1e-6, atol=1e-6)


def test_parameters_errors():
    known_parameters = ["p1", "p2"]
    values = np.array([1, 2])
    cov = np.array([[1.0, 0.5], [0.5, 1.0]])
    params = Parameters(known_parameters, values, cov)
    errors = params.errors
    expected_errors = np.array([1.0, 1.0])
    npt.assert_allclose(errors, expected_errors, rtol=1e-6, atol=1e-6)
    known_parameters = ["param1", "param2"]


def test_parameters_chi2():
    known_parameters = ["p1", "p2"]
    values = np.array([1, 2])
    cov = np.array([[1.0, 0.5], [0.5, 1.0]])
    params = Parameters(known_parameters, values, cov)
    chi2 = params.chi2
    expected_chi2 = 4
    npt.assert_allclose(chi2, expected_chi2, rtol=1e-6, atol=1e-6)


def test_parameters_class():
    known_parameters = ["param1", "param2"]
    values = np.array([1, 2])
    cov = np.array([[1.0, 0.5], [0.5, 1.0]])

    params = Parameters(known_parameters, values, cov)
    assert isinstance(params, Parameters)
    assert params.known_parameters == known_parameters
    assert list(params) == [("param1", 1), ("param2", 2)]
    assert np.allclose(params.cov, cov)
    assert list(params) == [("param1", 1), ("param2", 2)]


def test_grid_cov():
    jac = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    cov = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    result = grid_cov(jac, cov)
    expected = np.array([[228, 552, 876], [516, 1245, 1974], [804, 1938, 3072]])
    np.testing.assert_array_equal(result, expected)


def test_is_iterable():
    assert is_iterable([1, 2, 3]) is True
    assert is_iterable("abc") is False
    assert is_iterable(1) is False
    assert is_iterable((1, 2, 3)) is True
    assert is_iterable({"a": 1, "b": 2}) is True


def test_rearrange_covariance():
    original_order = {"a": 0, "b": 1, "c": 2, "d": 3}
    new_order = ["d", "c", "b", "a"]
    cov = np.array([[1, 2, 3, 4], [2, 5, 6, 7], [3, 6, 8, 9], [4, 7, 9, 10]])
    expected = np.array([[10, 9, 7, 4], [9, 8, 6, 3], [7, 6, 5, 2], [4, 3, 2, 1]])

    assert np.allclose(rearrange_covariance(original_order, new_order, cov), expected)


def test_interpolation_domain():
    zenith_test_dataset = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90], dtype=float)
    mock_spl = dict([(format_angle(z), []) for z in zenith_test_dataset])
    known_parameters = ["p1", "p2"]
    values = np.array([1, 2])
    cov = np.array([[1.0, 0.5], [0.5, 1.0]])
    params = Parameters(known_parameters, values, cov)
    entry = _FluxEntry(
        "mock", fl_spl=mock_spl, jac_spl=mock_spl, params=params, debug=0
    )
    # entry._zenith_deg_arr = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90])

    # Test for a single angle within the range
    assert entry._interpolation_domain(45) == (4, 5)

    # Test for multiple angles within the range
    assert np.all(entry._interpolation_domain(np.array([45.0, 51.0])) == (4, 6))

    # Test for an angle outside the range
    try:
        entry._interpolation_domain(-5)
    except ValueError as e:
        assert (
            str(e)
            == "Requested zenith angles must be within the range 0.0000 - 90.0000"
        )

    # Test for multiple angles outside the range
    try:
        entry._interpolation_domain(np.array([-5.0, 95.0]))
    except ValueError as e:
        assert (
            str(e)
            == "Requested zenith angles must be within the range 0.0000 - 90.0000"
        )

    # Test for unsorted angles
    try:
        entry._interpolation_domain(np.array([50.0, 45.0]))
    except ValueError as e:
        assert str(e) == "Requested angles must be sorted in ascending order."
