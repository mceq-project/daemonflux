"""Offline response-library compatibility and calibration identity checks."""

import pickle

import numpy as np
import pytest
from scipy.interpolate import UnivariateSpline

from daemonflux import Flux
from daemonflux.metadata import file_sha256


def library(tmp_path, cov=None, metadata=True):
    names = ["primary_pivot", "yield_a", "yield_b"]
    flux = {
        "generic": {"0.0000": {"numuflux": UnivariateSpline([0, 3], [2, 2], k=1, s=0)}}
    }
    jac = {
        "generic": {
            "0.0000": {
                p: {"numuflux": UnivariateSpline([0, 3], [v, v], k=1, s=0)}
                for p, v in zip(names, [1, 2, 3])
            }
        }
    }
    payload = [names, flux, jac, cov, np.eye(1)]
    if metadata:
        payload.append(
            {
                "schema_version": 1,
                "parameters": [
                    {
                        "name": p,
                        "group": "primary" if i == 0 else "hadronic",
                        "units": "fractional",
                    }
                    for i, p in enumerate(names)
                ],
            }
        )
    path = tmp_path / "splines.pkl"
    path.write_bytes(pickle.dumps(payload))
    return path, payload


def load(path, **kwargs):
    return Flux(spl_file=path, keep_old_revisions=True, debug=0, **kwargs)


def test_nominal_without_invented_prior(tmp_path):
    path, _ = library(tmp_path)
    f = load(path, use_calibration=False)
    np.testing.assert_allclose(f.flux([2.0, 3.0], 0, "numuflux"), np.exp(2))
    with pytest.raises(ValueError, match="no parameter covariance"):
        f.error([2.0], 0, "numuflux")
    with pytest.raises(ValueError, match="no parameter covariance"):
        f.flux([2.0], 0, "numuflux", params={"yield_a": 1})


def test_metadata_groups_ignore_names_and_order(tmp_path):
    path, _ = library(tmp_path, np.diag([0.09, 0.04, 0.01]))
    f = load(path, use_calibration=False)
    expected = np.exp(2) * np.sqrt(4 * 0.04 + 9 * 0.01)
    np.testing.assert_allclose(
        f.error([2.0], 0, "numuflux", only_hadronic=True), expected
    )
    excluded = load(path, use_calibration=False, exclude=["yield_a"])
    assert excluded.params.cov.shape == (2, 2)
    np.testing.assert_allclose(excluded.params.cov, np.diag([0.09, 0.01]))


def test_decorrelation_preserves_fractional_variances(tmp_path):
    path, _ = library(
        tmp_path, np.array([[0.09, 0, 0], [0, 0.04, 0.01], [0, 0.01, 0.01]])
    )
    f = load(path, use_calibration=False, uncorrelated_hadr_errors=True)
    np.testing.assert_allclose(f.params.cov, np.diag([0.09, 0.04, 0.01]))


def test_calibration_hash_and_reordered_covariance(tmp_path):
    path, payload = library(tmp_path)
    names = payload[0][::-1]
    calibration = {
        "params": {p: {"value": 0.01 * i} for i, p in enumerate(names)},
        "cov_params": names,
        "cov_matrix": np.diag([0.01, 0.04, 0.09]),
    }
    cal = tmp_path / "calibration.pkl"
    cal.write_bytes(pickle.dumps(calibration))
    with pytest.raises(ValueError, match="require calibration spline_sha256"):
        load(path, cal_file=cal)
    calibration["spline_sha256"] = file_sha256(path)
    cal.write_bytes(pickle.dumps(calibration))
    f = load(path, cal_file=cal)
    np.testing.assert_allclose(f.params.cov, np.diag([0.09, 0.04, 0.01]))
    np.testing.assert_allclose(f.params.values, [0.02, 0.01, 0])
    calibration["spline_sha256"] = "0" * 64
    cal.write_bytes(pickle.dumps(calibration))
    with pytest.raises(ValueError, match="does not match"):
        load(path, cal_file=cal)


def test_reject_bad_metadata_and_covariance(tmp_path):
    path, payload = library(tmp_path, np.eye(3))
    payload[5]["parameters"].reverse()
    path.write_bytes(pickle.dumps(payload))
    with pytest.raises(ValueError, match="order"):
        load(path, use_calibration=False)
    path, payload = library(tmp_path, np.diag([1, -1, 1]))
    with pytest.raises(ValueError, match="positive semidefinite"):
        load(path, use_calibration=False)


def test_legacy_five_item_payload(tmp_path):
    path, _ = library(tmp_path, np.eye(3), metadata=False)
    f = load(path, use_calibration=False)
    assert f.metadata is None
    assert np.isfinite(f.error([2.0], 0, "numuflux"))
