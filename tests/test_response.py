import json
import pathlib
import pickle

import numpy as np
import numpy.testing as npt
import pytest

from daemonflux import Flux
from daemonflux.response import (
    ResponseModel,
    ResponseTable,
    basis_sha256,
    read_response_files,
    write_response_file,
)

HERE = pathlib.Path(__file__).parent
PARAMS = [
    {"name": "pi+_31G", "group": "hadronic", "units": "fractional", "secondary": 211},
    {
        "name": "K-_2P",
        "group": "hadronic",
        "units": "fractional",
        "secondary": -321,
        "transform": "log",
    },
    {
        "name": "p_31G",
        "group": "hadronic",
        "units": "fractional",
        "secondary": 2212,
        "combination": "factor",
    },
    {
        "name": "n_31G",
        "group": "hadronic",
        "units": "fractional",
        "secondary": 2112,
        "combination": "factor",
    },
    {"name": "GSF_p_1", "group": "primary", "units": "sigma"},
]


def _species(rng, n=12, curvature=True):
    x = np.geomspace(1.0, 1e4, n)
    table = {
        "x": x,
        "value": 1e-2 * x ** rng.uniform(-0.2, 0.2),
        "jacobian": rng.uniform(-0.4, 0.4, (len(PARAMS), n)),
    }
    if curvature:
        table["curvature"] = rng.uniform(-0.2, 0.2, (len(PARAMS), n))
    return table


def _profiles(seed=1, names=("site",)):
    rng = np.random.default_rng(seed)
    return {
        name: {
            angle: {s: _species(rng) for s in ("mu+", "mu-", "numu", "antinumu")}
            for angle in ("0.0000", "60.0000")
        }
        for name in names
    }


def _write(tmp_path, name="a.h5", profiles=None, params=PARAMS, **kw):
    meta = {"parameters": params, "dataset_id": name.split(".")[0]}
    meta.update(kw)
    return tmp_path / name, write_response_file(
        tmp_path / name,
        meta,
        profiles or _profiles(),
        covariance=0.01 * np.eye(len(params)),
    )


def _calibration(path, values, cov, names=None):
    names = names or [p["name"] for p in PARAMS]
    cal = {
        "params": {
            n: {"value": v, "number": i} for i, (n, v) in enumerate(zip(names, values))
        },
        "cov_params": names,
        "cov_matrix": cov,
        "basis_sha256": basis_sha256(
            [
                dict(
                    p,
                    combination=p.get("combination", "additive"),
                    transform=p.get("transform", "linear"),
                )
                for p in PARAMS
            ]
        ),
        "calibration_id": "test_20261001a",
    }
    with open(path, "wb") as f:
        pickle.dump(cal, f)
    return path


def test_roundtrip_and_sha(tmp_path):
    path, sha = _write(tmp_path)
    lib = read_response_files([path], {str(path): sha})
    assert lib.names == [p["name"] for p in PARAMS]
    t = lib.profiles["site"]["0.0000"]["mu+"]
    ref = _profiles()["site"]["0.0000"]["mu+"]
    npt.assert_array_equal(t.value, ref["value"])
    npt.assert_array_equal(t.curvature, ref["curvature"])
    with pytest.raises(ValueError, match="sha256"):
        read_response_files([path], {str(path): "0" * 64})


def test_writer_rejects_bad_tables(tmp_path):
    bad = _profiles()
    bad["site"]["0.0000"]["mu+"]["value"][3] = -1
    with pytest.raises(ValueError, match="positive"):
        _write(tmp_path, profiles=bad)
    with pytest.raises(ValueError, match="log transform"):
        _write(tmp_path, params=[dict(PARAMS[4], transform="log")] + PARAMS[:4])


def test_linear_additive_model_is_legacy_formula():
    rng = np.random.default_rng(3)
    params = [
        {"name": f"q{i}", "group": "hadronic", "units": "fractional"} for i in range(4)
    ]
    t = ResponseTable(np.geomspace(1, 100, 8), np.ones(8), rng.normal(size=(4, 8)))
    theta = rng.normal(size=4) * 0.1
    flux, _ = ResponseModel(params).evaluate(t, t.log_x, theta)
    npt.assert_allclose(flux, 1 + theta @ t.jacobian, rtol=1e-14)


def test_model_combination_and_transform():
    rng = np.random.default_rng(4)
    tab = _species(rng)
    t = ResponseTable(tab["x"], tab["value"], tab["jacobian"], tab["curvature"])
    theta = np.array([0.1, -0.5, 0.2, -0.3, 1.5])
    flux, _ = ResponseModel(PARAMS).evaluate(t, t.log_x, theta)
    a = theta.copy()
    a[1] = np.expm1(theta[1])
    J, C = tab["jacobian"], tab["curvature"]
    term = J * a[:, None] + 0.5 * C * a[:, None] ** 2
    expected = tab["value"] * (1 + term[[0, 1, 4]].sum(0)) * (1 + term[2]) * (1 + term[3])
    npt.assert_allclose(flux, expected, rtol=1e-13)


def test_gradient_matches_finite_difference():
    rng = np.random.default_rng(5)
    tab = _species(rng)
    t = ResponseTable(tab["x"], tab["value"], tab["jacobian"], tab["curvature"])
    model = ResponseModel(PARAMS)
    theta = np.array([0.1, -0.5, 0.2, -0.3, 1.5])
    log_e = np.log(np.geomspace(1.3, 9e3, 17))
    _, grad = model.evaluate(t, log_e, theta, gradient=True)
    h = 1e-6
    for i in range(len(PARAMS)):
        up, dn = theta.copy(), theta.copy()
        up[i] += h
        dn[i] -= h
        fd = (model.evaluate(t, log_e, up)[0] - model.evaluate(t, log_e, dn)[0]) / (2 * h)
        npt.assert_allclose(grad[i], fd, rtol=1e-6, atol=1e-12)


def test_cubic_reproduces_nodes(tmp_path):
    rng = np.random.default_rng(6)
    tab = _species(rng)
    t = ResponseTable(tab["x"], tab["value"], tab["jacobian"], interpolation="cubic")
    f0, J, _ = t.interpolate(t.log_x)
    npt.assert_allclose(f0, tab["value"], rtol=1e-12)
    npt.assert_allclose(J, tab["jacobian"], rtol=1e-12, atol=1e-14)


def test_flux_derived_quantities_and_errors(tmp_path):
    a, _ = _write(tmp_path, "a.h5", _profiles(1, ("site_a",)))
    b, _ = _write(tmp_path, "b.h5", _profiles(2, ("site_b",)))
    cov = np.diag([0.01, 0.04, 0.0025, 0.0025, 1.0])
    values = [0.05, -0.2, 0.1, -0.1, 0.5]
    cal = _calibration(tmp_path / "cal.pkl", values, cov)
    flux = Flux(spl_file=[a, b], cal_file=cal, debug=0)
    assert flux.supported_fluxes == ["site_a", "site_b"]
    assert flux.calibration_info["calibration_id"] == "test_20261001a"
    E = np.geomspace(2.0, 5e3, 9)
    for site in ("site_a", "site_b"):
        mp = flux.flux(E, "0.0000", "mu+", exp=site)
        mm = flux.flux(E, "0.0000", "mu-", exp=site)
        npt.assert_allclose(
            flux.flux(E, "0.0000", "muflux", exp=site), mp + mm, rtol=1e-14
        )
        npt.assert_allclose(
            flux.flux(E, "0.0000", "muratio", exp=site), mp / mm, rtol=1e-14
        )
        entry = flux[site]
        for q in ("mu+", "muratio", "numuflux"):
            g = entry.gradient(E, "0.0000", q)
            npt.assert_allclose(
                flux.error(E, "0.0000", q, exp=site),
                np.sqrt(np.einsum("pe,pq,qe->e", g, cov, g)),
                rtol=1e-12,
            )
    # Zenith interpolation still works between tabulated angles.
    mid = flux.flux(E, 30.0, "muflux", exp="site_a")
    assert np.all(np.isfinite(mid)) and mid.shape == E.shape


def test_ratio_error_is_exact_propagation(tmp_path):
    """The ratio gradient equals R * (dlnF+ - dlnF-), so its error is exact."""
    a, _ = _write(tmp_path, "a.h5", _profiles(7, ("s",)))
    cov = np.diag([0.01, 0.04, 0.0025, 0.0025, 1.0])
    flux = Flux(
        spl_file=[a], cal_file=_calibration(tmp_path / "c.pkl", [0.0] * 5, cov), debug=0
    )
    E = np.geomspace(2.0, 5e3, 7)
    e = flux["s"]
    gp, gm = e.gradient(E, "0.0000", "mu+"), e.gradient(E, "0.0000", "mu-")
    fp, fm = flux.flux(E, "0.0000", "mu+"), flux.flux(E, "0.0000", "mu-")
    npt.assert_allclose(
        e.gradient(E, "0.0000", "muratio"), (gp * fm - fp * gm) / fm**2, rtol=1e-12
    )


def test_basis_and_profile_conflicts(tmp_path):
    a, _ = _write(tmp_path, "a.h5", _profiles(1, ("x",)))
    other = [dict(p) for p in PARAMS]
    other[2]["combination"] = "additive"
    b, _ = _write(tmp_path, "b.h5", _profiles(2, ("y",)), params=other)
    with pytest.raises(ValueError, match="basis"):
        Flux(spl_file=[a, b], use_calibration=False, debug=0)
    c, _ = _write(tmp_path, "c.h5", _profiles(3, ("x",)))
    with pytest.raises(ValueError, match="more than one file"):
        Flux(spl_file=[a, c], use_calibration=False, debug=0)


def test_calibration_basis_mismatch(tmp_path):
    a, _ = _write(tmp_path, "a.h5")
    cal = _calibration(tmp_path / "cal.pkl", [0.0] * 5, np.eye(5) * 0.01)
    with open(cal, "rb") as f:
        d = pickle.load(f)
    d["basis_sha256"] = "f" * 64
    with open(cal, "wb") as f:
        pickle.dump(d, f)
    with pytest.raises(ValueError, match="different parameter basis"):
        Flux(spl_file=[a], cal_file=cal, debug=0)


def test_bundle_verifies_files(tmp_path):
    _, sha = _write(tmp_path, "a.h5")
    cal = _calibration(tmp_path / "cal.pkl", [0.0] * 5, np.eye(5) * 0.01)
    from daemonflux.metadata import file_sha256

    manifest = {
        "format": "daemonflux-bundle",
        "bundle_id": "test",
        "files": [{"path": "a.h5", "sha256": sha}],
        "calibration": {"path": "cal.pkl", "sha256": file_sha256(cal)},
    }
    (tmp_path / "bundle.json").write_text(json.dumps(manifest))
    flux = Flux(bundle=tmp_path / "bundle.json", debug=0)
    assert flux.calibration_info["calibration_id"] == "test_20261001a"
    manifest["files"][0]["sha256"] = "0" * 64
    (tmp_path / "bundle.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="sha256"):
        Flux(bundle=tmp_path / "bundle.json", debug=0)


def test_legacy_conversion_is_lossless(tmp_path):
    from daemonflux.convert import convert_legacy

    legacy = HERE / "test_daemonsplines_generic_202303_1.pkl"
    cal = HERE / "test_calibration_default_202303_1.pkl"
    written = convert_legacy(legacy, tmp_path, build="test")
    old = Flux(spl_file=str(legacy), cal_file=str(cal), debug=0)
    new = Flux(spl_file=list(written), cal_file=str(cal), debug=0)
    E = np.geomspace(1.0, 1e5, 40)
    for exp in old.supported_fluxes:
        angle = old[exp].zenith_angles[0]
        for q in ("mu+", "numu", "antinue"):
            if q not in old[exp]._quantities:
                continue
            npt.assert_allclose(
                new.flux(E, angle, q, exp=exp), old.flux(E, angle, q, exp=exp), rtol=1e-12
            )
            npt.assert_allclose(
                new.error(E, angle, q, exp=exp),
                old.error(E, angle, q, exp=exp),
                rtol=1e-12,
            )
