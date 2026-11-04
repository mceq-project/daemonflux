import numpy as np
import numpy.testing as npt
import pytest
from test_response import PARAMS, _calibration

from daemonflux import Flux
from daemonflux.response import write_response_file
from daemonflux.utils import format_angle

AZ_EDGES = [0.0, 90.0, 180.0, 270.0, 360.0]
COS_EDGES = [-1.0, -0.5, 0.0, 0.5, 1.0]
CENTRES = [format_angle(np.degrees(np.arccos(c))) for c in (-0.75, -0.25, 0.25, 0.75)]
SPECIES = ("numu", "antinumu")
BLEND = [10.0, 50.0]
COV = np.diag([0.01, 0.04, 0.0025, 0.0025, 1.0])


def _table(rng, x):
    return {
        "x": x,
        "value": 1e-2 * x ** rng.uniform(-0.2, 0.2) * rng.uniform(0.5, 2.0),
        "jacobian": rng.uniform(-0.4, 0.4, (len(PARAMS), x.size)),
    }


def _sky_files(tmp_path, high_energy=True, seed=3):
    rng = np.random.default_rng(seed)
    low, high = np.geomspace(0.1, 100.0, 13), np.geomspace(1.0, 1e6, 19)
    names = [f"site_az{k:02d}" for k in range(len(AZ_EDGES) - 1)]
    maps = {
        n: {a: {s: _table(rng, low) for s in SPECIES} for a in CENTRES} for n in names
    }
    sky = {
        "site": {
            "profiles": names,
            "azimuth_edges_deg": AZ_EDGES,
            "azimuth_convention": "test",
            "cos_zenith_edges": COS_EDGES,
            "high_energy": {"profile": "site_1d", "blend_GeV": BLEND},
        }
    }
    meta = {"parameters": PARAMS}
    a = tmp_path / "map.h5"
    write_response_file(a, meta, maps, covariance=COV, sky=sky)
    files = [a]
    if high_energy:
        he_species = SPECIES + tuple("pr_" + s for s in SPECIES)
        one_d = {
            "site_1d": {a: {s: _table(rng, high) for s in he_species} for a in CENTRES}
        }
        b = tmp_path / "one_d.h5"
        write_response_file(b, meta, one_d, covariance=COV)
        files.append(b)
    cal = _calibration(tmp_path / "cal.pkl", [0.05, -0.2, 0.1, -0.1, 0.5], COV)
    return Flux(spl_file=files, cal_file=cal, debug=0), names


def _bin_mean(flux_fn, c_lo, c_hi, az_lo, az_hi, nodes=3):
    """Average of flux_fn(zenith_deg, azimuth_deg) over a bin, Gauss-Legendre in
    cos(zenith) and azimuth (exact for the piecewise-quadratic reconstruction)."""
    t, w = np.polynomial.legendre.leggauss(nodes)
    c = 0.5 * (c_lo + c_hi) + 0.5 * (c_hi - c_lo) * t
    az = 0.5 * (az_lo + az_hi) + 0.5 * (az_hi - az_lo) * t
    zen = np.degrees(np.arccos(c))
    values = flux_fn(zen[None, :], az[:, None])  # (n_E, n_az, n_zen)
    return np.einsum("eaz,a,z->e", values, w / 2, w / 2)


def test_bins_reproduce_tables(tmp_path):
    flux, names = _sky_files(tmp_path)
    assert flux.sky_sites == ["site"]
    sky = flux["site"]
    E = np.geomspace(0.1, 9.9, 7)
    for k, name in enumerate(names):
        for i, angle in enumerate(CENTRES):
            ref = flux.flux(E, angle, "numu", exp=name)
            # anywhere inside the bin, binned returns the bin average
            for dz, da in ((0.0, 0.0), (1.0, 30.0)):
                npt.assert_allclose(
                    sky.flux(
                        E,
                        float(angle) + dz,
                        "numu",
                        azimuth_deg=45.0 + 90.0 * k + da,
                        binned=True,
                    ),
                    ref,
                    rtol=1e-13,
                )
            # the reconstruction in the bin averages to the table
            mean = _bin_mean(
                lambda z, a: sky.flux(E, z, "numu", azimuth_deg=a),
                COS_EDGES[i],
                COS_EDGES[i + 1],
                AZ_EDGES[k],
                AZ_EDGES[k + 1],
            )
            npt.assert_allclose(mean, ref, rtol=1e-12)


def test_reconstruction_is_exact_for_quadratics(tmp_path):
    """Bin averages of a flux quadratic in cos(zenith) in each hemisphere (with a
    cusp at the horizon) and constant in azimuth give back the flux (from three
    bins per hemisphere on)."""
    edges = np.linspace(-1.0, 1.0, 9)
    centres = [
        format_angle(np.degrees(np.arccos(c))) for c in 0.5 * (edges[1:] + edges[:-1])
    ]
    shape = {-1: (1.0, 0.5, 2.0), 1: (1.0, -0.3, 1.5)}  # 1 + b c + a c^2 per side

    def g(c):
        a0, b, a = shape[1 if c >= 0 else -1]
        return a0 + b * c + a * c**2

    def g_mean(lo, hi):
        a0, b, a = shape[1 if lo >= 0 else -1]
        cube = (hi**3 - lo**3) / 3
        return a0 + (b * (hi**2 - lo**2) / 2 + a * cube) / (hi - lo)

    x = np.geomspace(0.1, 100.0, 13)
    jac = np.random.default_rng(1).uniform(-0.4, 0.4, (len(PARAMS), x.size))
    base = 1e-2 * x**0.1
    names = [f"site_az{k:02d}" for k in range(len(AZ_EDGES) - 1)]
    maps = {
        n: {
            a: {"numu": {"x": x, "value": base * g_mean(lo, hi), "jacobian": jac}}
            for a, lo, hi in zip(centres, edges[:-1], edges[1:])
        }
        for n in names
    }
    sky = {
        "site": {
            "profiles": names,
            "azimuth_edges_deg": AZ_EDGES,
            "cos_zenith_edges": edges,
        }
    }
    write_response_file(
        tmp_path / "q.h5", {"parameters": PARAMS}, maps, covariance=COV, sky=sky
    )
    site = Flux(spl_file=[tmp_path / "q.h5"], use_calibration=False, debug=0)["site"]
    E = np.array([0.3, 30.0])
    unit = site.flux(E, 170.0, "numu", binned=True) / g_mean(edges[0], edges[1])
    for c in (-1.0, -0.8, -0.31, -0.02, 0.0, 0.07, 0.5, 0.93, 1.0):
        expect = unit * g(c)  # at the horizon both one-sided values are 1
        npt.assert_allclose(
            site.flux(E, np.degrees(np.arccos(c)), "numu", azimuth_deg=[17.0, 250.0]),
            np.stack([expect, expect], axis=1),
            rtol=1e-12,
        )


def test_azimuth_reconstruction_and_average(tmp_path):
    flux, names = _sky_files(tmp_path)
    sky, E, zen = flux["site"], np.geomspace(0.2, 5.0, 4), float(CENTRES[2])
    f = [
        sky.flux(E, zen, "antinumu", azimuth_deg=45.0 + 90.0 * k, binned=True)
        for k in range(len(names))
    ]
    # periodic
    npt.assert_allclose(
        sky.flux(E, zen, "antinumu", azimuth_deg=-30.0),
        sky.flux(E, zen, "antinumu", azimuth_deg=330.0),
        rtol=1e-14,
    )
    npt.assert_allclose(
        sky.flux(E, zen, "antinumu", azimuth_deg=0.0),
        sky.flux(E, zen, "antinumu", azimuth_deg=360.0),
        rtol=1e-14,
    )
    # the average over azimuth is the width-weighted mean of the bins
    npt.assert_allclose(
        sky.flux(E, zen, "antinumu", binned=True), np.mean(f, axis=0), rtol=1e-13
    )
    az = np.linspace(0.0, 360.0, 721)[:-1]
    npt.assert_allclose(
        sky.flux(E, zen, "antinumu", azimuth_deg=az).mean(axis=1),
        sky.flux(E, zen, "antinumu"),
        rtol=1e-12,
    )
    npt.assert_allclose(
        flux.flux(E, zen, "antinumu", exp="site", azimuth_deg=90.0),
        sky.flux(E, zen, "antinumu", azimuth_deg=90.0),
    )
    # broadcasting of directions: energy first, then the direction shape
    out = sky.flux(E, [zen, 40.0], "antinumu", azimuth_deg=[[45.0], [135.0], [225.0]])
    assert out.shape == (E.size, 3, 2)
    npt.assert_allclose(out[:, 1, 1], sky.flux(E, 40.0, "antinumu", azimuth_deg=135.0))


def test_high_energy_join(tmp_path):
    (tmp_path / "joined").mkdir()
    (tmp_path / "map").mkdir()
    flux, _ = _sky_files(tmp_path / "joined")
    map_only = _sky_files(tmp_path / "map", high_energy=False)[0]["site"]
    sky = flux["site"]
    lower, upper = sky.energy_range("numu")
    assert (lower, upper) == (0.1, 1e6)
    zen, az = float(CENTRES[3]), 135.0
    one_d = flux["site_1d"]
    above = np.geomspace(50.0, 1e6, 6)
    npt.assert_allclose(
        sky.flux(above, [zen, 50.0], "numu", azimuth_deg=az),
        np.stack([one_d.flux(above, z, "numu") for z in (CENTRES[3], "50")], axis=1),
        rtol=1e-13,
    )
    mid = np.sqrt(BLEND[0] * BLEND[1])
    expect = 0.5 * (
        map_only.flux(mid, zen, "numu", azimuth_deg=az)
        + one_d.flux(mid, CENTRES[3], "numu")
    )
    npt.assert_allclose(sky.flux(mid, zen, "numu", azimuth_deg=az), expect, rtol=1e-13)
    # the gradient is joined the same way, so errors stay exact
    g = sky.gradient(mid, zen, "numuratio", azimuth_deg=az)
    npt.assert_allclose(
        sky.error(mid, zen, "numuratio", azimuth_deg=az),
        np.sqrt(np.einsum("pe,pq,qe->e", g, sky.covariance, g)),
        rtol=1e-12,
    )
    gm = map_only.gradient(mid, zen, "numu", azimuth_deg=az)
    gh = one_d.gradient(mid, CENTRES[3], "numu")
    npt.assert_allclose(
        sky.gradient(mid, zen, "numu", azimuth_deg=az), 0.5 * (gm + gh), rtol=1e-12
    )
    # binned: the bin average of the high-energy profile's linear interpolant
    lo, hi = COS_EDGES[2], COS_EDGES[3]
    c = np.linspace(lo, hi, 2001)
    values = np.stack(
        [one_d.flux(above, z, "numu") for z in np.degrees(np.arccos(c))], axis=1
    )
    npt.assert_allclose(
        sky.flux(above, float(CENTRES[2]), "numu", binned=True),
        0.5 * (values[:, 1:] + values[:, :-1]) @ np.diff(c) / (hi - lo),
        rtol=1e-6,
    )
    npt.assert_allclose(sky.covariance, COV)
    assert sky.parameter_names == [p["name"] for p in PARAMS]
    with pytest.raises(AssertionError, match="out of range"):
        sky.flux(2e6, zen, "numu")


def test_gradient_matches_parameter_shift(tmp_path):
    flux, _ = _sky_files(tmp_path)
    sky, E = flux["site"], np.array([0.5, 20.0, 1e3])
    zen, az = 70.0, 200.0
    g = sky.gradient(E, zen, "numuratio", azimuth_deg=az)
    base = sky.flux(E, zen, "numuratio", azimuth_deg=az)
    for i, name in enumerate(sky.parameter_names[:3]):
        eps = 1e-6
        shifted = sky.flux(E, zen, "numuratio", azimuth_deg=az, params={name: eps})
        sigma = np.sqrt(COV[i, i])
        npt.assert_allclose((shifted - base) / (eps * sigma), g[i], rtol=1e-4, atol=1e-10)


def test_without_high_energy_profile(tmp_path):
    flux, _ = _sky_files(tmp_path, high_energy=False)
    sky = flux["site"]
    assert sky.high_energy is None
    assert sky.energy_range("numu") == (0.1, 100.0)
    with pytest.raises(AssertionError, match="load site_1d"):
        sky.flux(200.0, 30.0, "numu")
    npt.assert_allclose(sky.flux(80.0, 30.0, "numu"), sky.flux(80.0, 30.0, "numu"))


def test_azimuth_needs_a_sky_map(tmp_path):
    flux, names = _sky_files(tmp_path)
    with pytest.raises(ValueError, match="no azimuth dependence"):
        flux.flux(1.0, CENTRES[0], "numu", exp=names[0], azimuth_deg=10.0)
    with pytest.raises(ValueError, match="0-180"):
        flux["site"].flux(1.0, 181.0, "numu")


def test_sky_description_is_checked(tmp_path):
    rng = np.random.default_rng(0)
    x = np.geomspace(1.0, 10.0, 5)
    profiles = {"p": {CENTRES[0]: {"numu": _table(rng, x)}}}
    good = {
        "profiles": ["p"],
        "azimuth_edges_deg": [0, 360],
        "cos_zenith_edges": [-1, -0.5],
    }
    for bad, match in (
        (dict(good, azimuth_edges_deg=[0, 180]), "360"),
        (dict(good, profiles=["p", "q"]), "one profile per azimuth bin"),
        (dict(good, cos_zenith_edges=[0.5, -0.5]), "cos"),
        (dict(good, high_energy={"profile": "q", "blend_GeV": [5, 2]}), "lower < upper"),
    ):
        with pytest.raises(ValueError, match=match):
            write_response_file(
                tmp_path / "s.h5", {"parameters": PARAMS}, profiles, sky={"s": bad}
            )
    write_response_file(
        tmp_path / "s.h5",
        {"parameters": PARAMS},
        profiles,
        covariance=COV,
        sky={"s": good},
    )
    sky = Flux(spl_file=[tmp_path / "s.h5"], use_calibration=False, debug=0)["s"]
    npt.assert_allclose(
        sky.flux(5.0, 140.0, "numu"), sky.flux(5.0, float(CENTRES[0]), "numu"), rtol=1e-14
    )


def test_prompt_from_the_high_energy_profile(tmp_path):
    flux, _ = _sky_files(tmp_path)
    sky, one_d = flux["site"], flux["site_1d"]
    assert {"pr_numu", "total_numu", "total_numuflux", "total_numuratio"} <= set(
        sky.quantities
    )
    assert sky.energy_range("total_numu") == (1.0, 1e6)
    E, zen = np.geomspace(1.0, 1e6, 9), 120.0
    for az in (10.0, 250.0):
        npt.assert_allclose(
            sky.flux(E, zen, "pr_numu", azimuth_deg=az),
            one_d.flux(E, zen, "pr_numu"),
            rtol=1e-13,
        )
        npt.assert_allclose(
            sky.flux(E, zen, "total_numu", azimuth_deg=az),
            sky.flux(E, zen, "numu", azimuth_deg=az) + one_d.flux(E, zen, "pr_numu"),
            rtol=1e-13,
        )
    g = sky.gradient(E, zen, "total_numuratio", azimuth_deg=30.0)
    npt.assert_allclose(
        sky.error(E, zen, "total_numuratio", azimuth_deg=30.0),
        np.sqrt(np.einsum("pe,pq,qe->e", g, COV, g)),
        rtol=1e-12,
    )
