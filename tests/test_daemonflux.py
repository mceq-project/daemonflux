import numpy as np
import numpy.testing as npt
from daemonflux.flux import Parameters, _FluxEntry, Flux
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
    # entry._zenith_deg_arr = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90])

    # Test for a single angle within the range
    assert entry._interpolation_domain(45) == (4, 6)

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


# # Creating the test splines from one of the files
# import pickle

# (known_pars, _fl_spl, _jac_spl, cov) = pickle.load(
#     open("../daemonsplines_generic_20230207.pkl", "rb")
# )

# test_fl_spl = {}
# test_fl_spl["generic"] = {}
# test_jac_spl = {}
# test_jac_spl["generic"] = {}
# test_known_pars = known_pars[:3] + ["GSF_1", "GSF_2"]
# for a in ["0.0000", "18.1949"]:
#     test_fl_spl["generic"][a] = {}
#     test_jac_spl["generic"][a] = {}
#     for kp in test_known_pars:
#         test_jac_spl["generic"][a][kp] = {}
# for a in ["0.0000", "18.1949"]:
#     test_fl_spl["generic"][a]["numuflux"] = _fl_spl["generic"][a]["numuflux"]
#     test_fl_spl["generic"][a]["muflux"] = _fl_spl["generic"][a]["muflux"]
#     for kp in test_known_pars:
#         test_jac_spl["generic"][a][kp] = {}
#         test_jac_spl["generic"][a][kp]["numuflux"] = _jac_spl[
#           "generic"][a][kp]["numuflux"]
#         test_jac_spl["generic"][a][kp]["muflux"] = _jac_spl[
#           "generic"][a][kp]["muflux"]

# new_values = []
# keep_cov = []
# for ik, k in enumerate(known_pars):
#     if k in test_known_pars:
#         keep_cov.append(ik)

# test_cov = np.take(np.take(cov, keep_cov, axis=0), keep_cov, axis=1)

# pickle.dump(
#     [test_known_pars, test_fl_spl, test_jac_spl, test_cov],
#     open("../tests/test_daemonsplines_generic_20230207.pkl", "wb"),
#     protocol=-1,
# )
# Building the covariance matrix
# par_idx_map = {p: i for i, p in enumerate(fl_ic_nc.params.known_parameters[:-6])}
# fl_ic_nc.params.cov[par_idx_map['pi-_158G'],par_idx_map['pi-_2P'] ]
# par_idx_map.keys()
# correlations = {
#     "K+_158G": ["K+_2P"],
#     "K-_158G": ["K-_2P"],
#     "p_158G": ["p_2P"],
#     "n_158G": ["n_2P"],
#     "pi+_158G": ["pi+_20T", "pi+_2P"],
#     "pi-_158G": ["pi-_20T", "pi-_2P"],
#     "pi+_20T": ["pi+_158G", "pi+_2P"],
#     "pi-_20T": ["pi-_158G", "pi-_2P"],
# }

# par_idx_map = {p: i for i, p in enumerate(fl_ic_nc.params.known_parameters[:-6])}
# cov = np.diag(len(fl_ic_nc.params.known_parameters[:-6]) * [1.0])
# for ip, p in enumerate(fl_ic_nc.params.known_parameters[:-6]):
#     for q in correlations.get(p, []):
#         cov[ip, par_idx_map[q]] = 1.0
#         cov[par_idx_map[q], ip] = 1.0
# plt.spy(cov)

# plt.gca().set_xticks(np.arange(0.5, len(fl_ic_nc.params.known_parameters[:-6])+0.5,1))
# plt.gca().set_xticklabels(fl_ic_nc.params.known_parameters[:-6],
#   rotation=45, fontsize=8)
# plt.gca().set_yticks(np.arange(0.5, len(fl_ic_nc.params.known_parameters[:-6])+0.5,1))
# plt.gca().set_yticklabels(fl_ic_nc.params.known_parameters[:-6],
#   rotation=45, fontsize=8)


def test_Flux():
    import pathlib

    basep = pathlib.Path(__file__).parent.absolute()
    fl_test = Flux(
        "",
        spl_file=basep / "test_daemonsplines_generic_20230207.pkl",
        cal_file=basep / "test_calibration_20230207.pkl",
        use_calibration=True,
        debug=1,
    )
    fl_test_nc = Flux(
        "",
        spl_file=basep / "test_daemonsplines_generic_20230207.pkl",
        use_calibration=False,
        debug=1,
    )
    egrid = np.logspace(0, 8)

    assert np.allclose(np.sum(fl_test.flux(egrid, "0.0000", "numuflux")), 0.786210673)
    assert np.allclose(np.sum(fl_test.flux(egrid, "18.1949", "numuflux")), 0.804512178)
    assert np.allclose(np.sum(fl_test.flux(egrid, 10, "numuflux")), 0.79177147)
    assert np.allclose(np.sum(fl_test.flux(egrid, "0.0000", "muflux")), 3.238592629)
    assert np.allclose(np.sum(fl_test.flux(egrid, "18.1949", "muflux")), 3.279707715)
    assert np.allclose(np.sum(fl_test.flux(egrid, 10, "muflux")), 3.25108520)

    assert np.allclose(np.sum(fl_test_nc.flux(egrid, "0.0000", "numuflux")), 0.7803695)
    assert np.allclose(np.sum(fl_test_nc.flux(egrid, "18.1949", "numuflux")), 0.7977987)
    assert np.allclose(np.sum(fl_test_nc.flux(egrid, 10, "numuflux")), 0.7856653)
    assert np.allclose(np.sum(fl_test_nc.flux(egrid, "0.0000", "muflux")), 3.1390889)
    assert np.allclose(np.sum(fl_test_nc.flux(egrid, "18.1949", "muflux")), 3.176512)
    assert np.allclose(np.sum(fl_test_nc.flux(egrid, 10, "muflux")), 3.1504598)

    # Test that fluxes are different when using calibration
    assert np.sum(fl_test.flux(egrid, "0.0000", "numuflux")) != np.sum(
        fl_test_nc.flux(egrid, "0.0000", "numuflux")
    )
    assert np.sum(fl_test.flux(egrid, "18.1949", "numuflux")) != np.sum(
        fl_test_nc.flux(egrid, "18.1949", "numuflux")
    )
    assert np.sum(fl_test.flux(egrid, 10, "numuflux")) != np.sum(
        fl_test_nc.flux(egrid, 10, "numuflux")
    )
    assert np.sum(fl_test.flux(egrid, "0.0000", "muflux")) != np.sum(
        fl_test_nc.flux(egrid, "0.0000", "muflux")
    )
    assert np.sum(fl_test.flux(egrid, "18.1949", "muflux")) != np.sum(
        fl_test_nc.flux(egrid, "18.1949", "muflux")
    )
    assert np.sum(fl_test.flux(egrid, 10, "muflux")) != np.sum(
        fl_test_nc.flux(egrid, 10, "muflux")
    )


def test_Flux_error():
    import pathlib

    basep = pathlib.Path(__file__).parent.absolute()
    fl_test = Flux(
        "",
        spl_file=basep / "test_daemonsplines_generic_20230207.pkl",
        cal_file=basep / "test_calibration_20230207.pkl",
        use_calibration=True,
        debug=1,
    )
    fl_test_nc = Flux(
        "",
        spl_file=basep / "test_daemonsplines_generic_20230207.pkl",
        use_calibration=False,
        debug=1,
    )
    egrid = np.logspace(0, 8)

    assert np.allclose(np.sum(fl_test.error(egrid, "0.0000", "numuflux")), 0.0395929433)
    assert np.allclose(
        np.sum(fl_test.error(egrid, "18.1949", "numuflux")), 0.0405816962
    )
    assert np.allclose(np.sum(fl_test.error(egrid, 10, "numuflux")), 0.03989337000)
    assert np.allclose(np.sum(fl_test.error(egrid, "0.0000", "muflux")), 0.0878357001)
    assert np.allclose(np.sum(fl_test.error(egrid, "18.1949", "muflux")), 0.0900900643)
    assert np.allclose(np.sum(fl_test.error(egrid, 10, "muflux")), 0.088520675)

    assert np.allclose(
        np.sum(fl_test_nc.error(egrid, "0.0000", "numuflux")), 0.029106047
    )
    assert np.allclose(
        np.sum(fl_test_nc.error(egrid, "18.1949", "numuflux")), 0.0296280658
    )
    assert np.allclose(np.sum(fl_test_nc.error(egrid, 10, "numuflux")), 0.0292646598)
    assert np.allclose(np.sum(fl_test_nc.error(egrid, "0.0000", "muflux")), 0.077694126)
    assert np.allclose(
        np.sum(fl_test_nc.error(egrid, "18.1949", "muflux")), 0.079532361
    )
    assert np.allclose(np.sum(fl_test_nc.error(egrid, 10, "muflux")), 0.078252663)

    # Test that errors are different depending on whether calibration is used
    assert np.sum(fl_test.error(egrid, "0.0000", "numuflux")) != np.sum(
        fl_test_nc.error(egrid, "0.0000", "numuflux")
    )
    assert np.sum(fl_test.error(egrid, "18.1949", "numuflux")) != np.sum(
        fl_test_nc.error(egrid, "18.1949", "numuflux")
    )
    assert np.sum(fl_test.error(egrid, 10, "numuflux")) != np.sum(
        fl_test_nc.error(egrid, 10, "numuflux")
    )
    assert np.sum(fl_test.error(egrid, "0.0000", "muflux")) != np.sum(
        fl_test_nc.error(egrid, "0.0000", "muflux")
    )
    assert np.sum(fl_test.error(egrid, "18.1949", "muflux")) != np.sum(
        fl_test_nc.error(egrid, "18.1949", "muflux")
    )
    assert np.sum(fl_test.error(egrid, 10, "muflux")) != np.sum(
        fl_test_nc.error(egrid, 10, "muflux")
    )


def test_default_url():
    import requests
    import pathlib

    basep = pathlib.Path(__file__).parent.absolute()
    fl_test = Flux(
        "",
        spl_file=basep / "test_daemonsplines_generic_20230207.pkl",
        cal_file=basep / "test_calibration_20230207.pkl",
        use_calibration=False,
        debug=1,
    )
    # test that the default url is reached

    url_generic_spl = (
        fl_test._default_url + fl_test._default_spl_file.format("generic") + ".zip"
    )
    url_cal = fl_test._default_url + fl_test._default_cal_file + ".zip"
    assert requests.head(url_generic_spl).status_code in [200, 302]
    assert requests.head(url_cal).status_code in [200, 302]
