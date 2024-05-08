import numpy as np
import numpy.testing as npt
import pytest
import pathlib

from daemonflux.flux import Flux, Parameters, _FluxEntry
from daemonflux.utils import format_angle, grid_cov, is_iterable, rearrange_covariance


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
    values = np.zeros(2)
    cov = np.array([[1.0, 0.5], [0.5, 1.0]])
    params = Parameters(known_parameters, values, cov)
    params.values = np.array([1, 2])
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
    zenith_test_dataset = np.array(
        [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100], dtype=float
    )
    mock_spl = dict(
        [(format_angle(z), {"numuflux": [], "muflux": []}) for z in zenith_test_dataset]
    )
    known_parameters = ["p1", "p2"]
    values = np.array([1, 2])
    cov = np.array([[1.0, 0.5], [0.5, 1.0]])
    params = Parameters(known_parameters, values, cov)
    entry = _FluxEntry(
        "mock", fl_spl=mock_spl, jac_spl=mock_spl, params=params, debug=0
    )

    # Test for a single angle within the range
    assert entry._interpolation_domain(45) == (4, 6)

    # Test for multiple angles within the range
    assert np.all(entry._interpolation_domain(np.array([45.0, 51.0])) == (4, 7))

    # Test for an angle outside the range
    try:
        entry._interpolation_domain(-5)
    except ValueError as e:
        assert (
            str(e)
            == "Requested zenith angles must be within the range 0.0000 - 100.0000"
        )

    # Test for multiple angles outside the range
    try:
        entry._interpolation_domain(np.array([-5.0, 105.0]))
    except ValueError as e:
        assert (
            str(e)
            == "Requested zenith angles must be within the range 0.0000 - 100.0000"
        )

    # Test for unsorted angles
    try:
        entry._interpolation_domain(np.array([50.0, 45.0]))
    except ValueError as e:
        assert str(e) == "Requested angles must be sorted in ascending order."


@pytest.fixture(scope="session")
def test_flux_calibrated():
    basep = pathlib.Path(__file__).parent.absolute()
    return Flux(
        "",
        spl_file=basep / "test_daemonsplines_generic_202303_1.pkl",
        cal_file=basep / "test_calibration_default_202303_1.pkl",
        use_calibration=True,
        debug=1,
    )


@pytest.fixture(scope="session")
def test_flux_not_calibrated():
    basep = pathlib.Path(__file__).parent.absolute()
    return Flux(
        "",
        spl_file=basep / "test_daemonsplines_generic_202303_1.pkl",
        cal_file=basep / "test_calibration_default_202303_1.pkl",
        use_calibration=False,
        debug=1,
    )


# Generate updated numbers for this test by running the following:
def test_Flux(test_flux_calibrated, test_flux_not_calibrated):
    egrid = np.logspace(0, 8)

    assert np.allclose(
        np.sum(test_flux_calibrated.flux(egrid, "0.0000", "numuflux")), 0.795157
    )
    assert np.allclose(
        np.sum(test_flux_calibrated.flux(egrid, "18.1949", "numuflux")), 0.813709
    )
    assert np.allclose(
        np.sum(test_flux_calibrated.flux(egrid, 10, "numuflux")), 0.800793
    )
    assert np.allclose(
        np.sum(test_flux_calibrated.flux(egrid, "0.0000", "muflux")), 3.248550
    )
    assert np.allclose(
        np.sum(test_flux_calibrated.flux(egrid, "18.1949", "muflux")), 3.289930
    )
    assert np.allclose(np.sum(test_flux_calibrated.flux(egrid, 10, "muflux")), 3.261123)
    assert np.allclose(
        np.sum(test_flux_not_calibrated.flux(egrid, "0.0000", "numuflux")), 0.780370
    )
    assert np.allclose(
        np.sum(test_flux_not_calibrated.flux(egrid, "18.1949", "numuflux")), 0.797799
    )
    assert np.allclose(
        np.sum(test_flux_not_calibrated.flux(egrid, 10, "numuflux")), 0.785665
    )
    assert np.allclose(
        np.sum(test_flux_not_calibrated.flux(egrid, "0.0000", "muflux")), 3.139089
    )
    assert np.allclose(
        np.sum(test_flux_not_calibrated.flux(egrid, "18.1949", "muflux")), 3.176512
    )
    assert np.allclose(
        np.sum(test_flux_not_calibrated.flux(egrid, 10, "muflux")), 3.150460
    )


def test_cal_active_on_flux(test_flux_calibrated, test_flux_not_calibrated):
    egrid = np.logspace(0, 8)

    # Test that fluxes are different when using calibration
    assert np.sum(test_flux_calibrated.flux(egrid, "0.0000", "numuflux")) != np.sum(
        test_flux_not_calibrated.flux(egrid, "0.0000", "numuflux")
    )
    assert np.sum(test_flux_calibrated.flux(egrid, "18.1949", "numuflux")) != np.sum(
        test_flux_not_calibrated.flux(egrid, "18.1949", "numuflux")
    )
    assert np.sum(test_flux_calibrated.flux(egrid, 10, "numuflux")) != np.sum(
        test_flux_not_calibrated.flux(egrid, 10, "numuflux")
    )
    assert np.sum(test_flux_calibrated.flux(egrid, "0.0000", "muflux")) != np.sum(
        test_flux_not_calibrated.flux(egrid, "0.0000", "muflux")
    )
    assert np.sum(test_flux_calibrated.flux(egrid, "18.1949", "muflux")) != np.sum(
        test_flux_not_calibrated.flux(egrid, "18.1949", "muflux")
    )
    assert np.sum(test_flux_calibrated.flux(egrid, 10, "muflux")) != np.sum(
        test_flux_not_calibrated.flux(egrid, 10, "muflux")
    )


def test_Flux_error(test_flux_calibrated, test_flux_not_calibrated):
    egrid = np.logspace(0, 8)

    assert np.allclose(
        np.sum(test_flux_calibrated.error(egrid, "0.0000", "numuflux")), 0.0322574
    )
    assert np.allclose(
        np.sum(test_flux_calibrated.error(egrid, "18.1949", "numuflux")), 0.0330159
    )
    assert np.allclose(
        np.sum(test_flux_calibrated.error(egrid, 10, "numuflux")), 0.0324878
    )
    assert np.allclose(
        np.sum(test_flux_calibrated.error(egrid, "0.0000", "muflux")), 0.0831389
    )
    assert np.allclose(
        np.sum(test_flux_calibrated.error(egrid, "18.1949", "muflux")), 0.0852520
    )
    assert np.allclose(
        np.sum(test_flux_calibrated.error(egrid, 10, "muflux")), 0.0837810
    )
    assert np.allclose(
        np.sum(test_flux_not_calibrated.error(egrid, "0.0000", "numuflux")), 0.0291060
    )
    assert np.allclose(
        np.sum(test_flux_not_calibrated.error(egrid, "18.1949", "numuflux")), 0.0296281
    )
    assert np.allclose(
        np.sum(test_flux_not_calibrated.error(egrid, 10, "numuflux")), 0.0292647
    )
    assert np.allclose(
        np.sum(test_flux_not_calibrated.error(egrid, "0.0000", "muflux")), 0.0776941
    )
    assert np.allclose(
        np.sum(test_flux_not_calibrated.error(egrid, "18.1949", "muflux")), 0.0795324
    )
    assert np.allclose(
        np.sum(test_flux_not_calibrated.error(egrid, 10, "muflux")), 0.0782527
    )


def test_cal_active_on_errors(test_flux_calibrated, test_flux_not_calibrated):
    # Test that errors are different depending on whether calibration is used
    egrid = np.logspace(0, 8)

    assert np.sum(test_flux_calibrated.error(egrid, "0.0000", "numuflux")) != np.sum(
        test_flux_not_calibrated.error(egrid, "0.0000", "numuflux")
    )
    assert np.sum(test_flux_calibrated.error(egrid, "18.1949", "numuflux")) != np.sum(
        test_flux_not_calibrated.error(egrid, "18.1949", "numuflux")
    )
    assert np.sum(test_flux_calibrated.error(egrid, 10, "numuflux")) != np.sum(
        test_flux_not_calibrated.error(egrid, 10, "numuflux")
    )
    assert np.sum(test_flux_calibrated.error(egrid, "0.0000", "muflux")) != np.sum(
        test_flux_not_calibrated.error(egrid, "0.0000", "muflux")
    )
    assert np.sum(test_flux_calibrated.error(egrid, "18.1949", "muflux")) != np.sum(
        test_flux_not_calibrated.error(egrid, "18.1949", "muflux")
    )
    assert np.sum(test_flux_calibrated.error(egrid, 10, "muflux")) != np.sum(
        test_flux_not_calibrated.error(egrid, 10, "muflux")
    )


# Write test function testing muflux is equal to sum of mu+ and mu- fluxes
def test_muflux_sums(test_flux_calibrated):
    grid = np.logspace(1, 5)
    for zen in test_flux_calibrated.zenith_angles:
        assert np.allclose(
            test_flux_calibrated.flux(grid, zen, "muflux"),
            test_flux_calibrated.flux(grid, zen, "mu+")
            + test_flux_calibrated.flux(grid, zen, "mu-"),
        ), "muflux should be equal to sum of mu+ and mu- fluxes"
        assert np.allclose(
            test_flux_calibrated.flux(grid, zen, "total_muflux"),
            test_flux_calibrated.flux(grid, zen, "total_mu+")
            + test_flux_calibrated.flux(grid, zen, "total_mu-"),
        ), "total_muflux should be equal to sum of total_mu+ and total_mu- fluxes"


# Also test the same for numuflux and nueflux
def test_numuflux_sums(test_flux_calibrated):
    grid = np.logspace(1, 5)
    for zen in test_flux_calibrated.zenith_angles:
        assert np.allclose(
            test_flux_calibrated.flux(grid, zen, "numuflux"),
            test_flux_calibrated.flux(grid, zen, "numu")
            + test_flux_calibrated.flux(grid, zen, "antinumu"),
            rtol=1e5,
        ), "numuflux should be equal to sum of numu and antinumu fluxes"
        assert np.allclose(
            test_flux_calibrated.flux(grid, zen, "total_numuflux"),
            test_flux_calibrated.flux(grid, zen, "total_numu")
            + test_flux_calibrated.flux(grid, zen, "total_antinumu"),
            rtol=1e5,
        ), "total_numuflux should be equal to sum of total_numu and total_antinumu"


# Also test the same for numuflux and nueflux
def test_nueflux_sums(test_flux_calibrated):
    grid = np.logspace(1, 5)
    for zen in test_flux_calibrated.zenith_angles:
        assert np.allclose(
            test_flux_calibrated.flux(grid, zen, "nueflux"),
            test_flux_calibrated.flux(grid, zen, "nue")
            + test_flux_calibrated.flux(grid, zen, "antinue"),
            rtol=1e5,
        ), "nueflux should be equal to sum of nue and antinue fluxes"
        assert np.allclose(
            test_flux_calibrated.flux(grid, zen, "total_nueflux"),
            test_flux_calibrated.flux(grid, zen, "total_nue")
            + test_flux_calibrated.flux(grid, zen, "total_antinue"),
            rtol=1e5,
        ), "total_nueflux should be equal to sum of total_nue and total_antinue fluxes"


def test_default_url(test_flux_calibrated):
    # test that the default url is reached
    from urllib import request

    url_generic_spl = (
        test_flux_calibrated._default_url
        + test_flux_calibrated._default_spl_file.format(
            location="generic", rev=test_flux_calibrated._revision
        )
        + ".zip"
    )
    assert test_flux_calibrated._revision == "202303_2"
    assert request.urlopen(url_generic_spl).status in [200, 302]

    for cal_set in ["default", "with_deis"]:

        url_cal = (
            test_flux_calibrated._default_url
            + test_flux_calibrated._default_cal_file.format(
                cset=cal_set, rev=test_flux_calibrated._revision
            )
            + ".zip"
        )
        assert request.urlopen(url_cal).status in [200, 302]


def test_chi2(test_flux_calibrated, test_flux_not_calibrated):
    assert test_flux_calibrated.chi2() == test_flux_not_calibrated.chi2() == 0.0
    params = test_flux_calibrated.params.known_parameters[:3]
    values = np.array([1, 2, 3])
    param_dict = dict(zip(params, values))
    assert test_flux_calibrated.chi2(param_dict) > 0
    assert test_flux_not_calibrated.chi2(param_dict)


def test_uncorrelated_errors():
    basep = pathlib.Path(__file__).parent.absolute()
    fl = Flux(
        "",
        spl_file=basep / "test_daemonsplines_generic_202303_1.pkl",
        cal_file=basep / "test_calibration_default_202303_1.pkl",
        use_calibration=False,
        uncorrelated_hadr_errors=True,
        debug=1,
    )
    assert np.allclose(
        fl.params.cov[: fl.params._n_non_gsf, : fl.params._n_non_gsf],
        np.diag(np.ones(fl.params._n_non_gsf)),
    )


def test_scalar_energy_arg(test_flux_calibrated):
    assert test_flux_calibrated.flux(100.0, 15.0, "numuflux").shape == ()
    assert test_flux_calibrated.error(100.0, 15.0, "numuflux").shape == ()
