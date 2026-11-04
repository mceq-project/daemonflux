"""Sky maps: fluxes by zenith and azimuth, joined to a 1D profile at high energy.

A sky map (``/sky/<site>`` in a response file, see ``response.check_sky``)
groups one profile per azimuth bin, each tabulated at the zenith-bin centres
with bin-averaged values. Above an energy window the map hands over to an
ordinary profile of point values without azimuth dependence, typically the 1D
cascade calculation for the same site, where geomagnetic and 3D effects have
vanished. Only the tables are stored; the join happens here.

The flux in a direction is reconstructed from the bin averages by a
histospline: the derivative of the cubic spline through the cumulative bin
integrals, which reproduces every bin average and, from three bins per side
on, is exact for fluxes quadratic in cos(zenith). In cos(zenith) the spline is
"not-a-knot" and split at the horizon, where the flux has a cusp; in azimuth it
is periodic. ``binned=True`` returns the bin average instead. The high-energy
profile is interpolated linearly in cos(zenith), as any other profile. Every
value is a weighted sum of table entries, so gradients, errors and ratios
between directions are exact.
"""

from itertools import pairwise

import numpy as np
from scipy.interpolate import CubicSpline

from .flux import _ResponseEntry
from .response import default_derived
from .utils import format_angle


def _smoothstep(log_e, lower, upper):
    """Weight of the high-energy profile: 0 below ``lower``, 1 above ``upper``."""
    t = np.clip((log_e - np.log(lower)) / (np.log(upper) - np.log(lower)), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _histospline(edges, periodic=False):
    """Point values from bin averages: x -> weights (len(x), n_bins).

    The weights are the derivative of the cubic spline through the cumulative
    integrals at ``edges``, one spline per unit vector of bin averages.
    Outside the edges the value at the nearest edge is kept.
    """
    width = np.diff(edges)
    if width.size == 1:
        return lambda x: np.ones((np.size(x), 1))
    cumulative = np.vstack([np.zeros(width.size), np.cumsum(np.diag(width), axis=0)])
    if not periodic:
        slope = CubicSpline(edges, cumulative).derivative()
        return lambda x: slope(np.clip(x, edges[0], edges[-1]))
    period = edges[-1] - edges[0]
    mean = width / period
    cumulative -= np.outer(edges - edges[0], mean)
    cumulative[-1] = cumulative[0]
    slope = CubicSpline(edges, cumulative, bc_type="periodic").derivative()
    return lambda x: slope((x - edges[0]) % period + edges[0]) + mean


def _zenith_histospline(edges):
    """``_histospline`` in cos(zenith), split at the horizon if it is an edge.

    At cos(zenith) = 0 the two one-sided values are averaged.
    """
    split = np.flatnonzero(np.isclose(edges, 0.0, atol=1e-12))
    if split.size == 0 or split[0] in (0, edges.size - 1):
        return _histospline(edges)
    j = int(split[0])
    lower, upper = _histospline(edges[: j + 1]), _histospline(edges[j:])

    def weights(c):
        c = np.asarray(c, dtype=float)
        w = np.zeros((c.size, edges.size - 1))
        below, above = c <= 0.0, c >= 0.0
        w[below, :j] = lower(c[below])
        w[above, j:] += upper(c[above])
        w[below & above] *= 0.5
        return w

    return weights


def _linear_weights(nodes, x):
    """Weights (len(x), len(nodes)) of linear interpolation, constant outside."""
    eye = np.eye(nodes.size)
    return np.stack([np.interp(x, nodes, row) for row in eye], axis=1)


def _linear_bin_averages(nodes, edges):
    """Weights (n_bins, len(nodes)): bin averages of the linear interpolant."""
    rows = []
    for a, b in pairwise(edges):
        x = np.unique(np.concatenate([[a, b], nodes[(nodes > a) & (nodes < b)]]))
        y = _linear_weights(nodes, x)
        rows.append(0.5 * np.diff(x) @ (y[1:] + y[:-1]) / (b - a))
    return np.array(rows)


class _SkyEntry(_ResponseEntry):
    """Evaluator of a sky map: ``flux(E, zenith, quantity, azimuth_deg=...)``.

    Zenith angles run from 0 (down-going) to 180 degrees (up-going); the
    azimuth follows the convention in ``azimuth_convention``. Without an
    azimuth the flux is averaged over azimuth.
    """

    def __init__(self, site, description, library, model, params, debug) -> None:
        missing = [p for p in description["profiles"] if p not in library.profiles]
        if missing:
            raise ValueError(f"Sky {site}: profiles {missing} are not loaded")
        self.label = site
        self.description = description
        self._model = model
        self._params = params
        self._debug = debug
        self._height_data = None
        self._linear_quantities = None
        self._index = {name: i for i, name in enumerate(model.names)}
        # Attributes of the base classes that do not apply to a sky map.
        self._tables, self._fl_spl, self._jac_spl = {}, {}, {}
        self._spl_contains_average = False

        maps = [library.profiles[p] for p in description["profiles"]]
        self._az_edges = np.asarray(description["azimuth_edges_deg"], dtype=float)
        self._cos_edges = np.asarray(description["cos_zenith_edges"], dtype=float)
        keys = [a for a in maps[0] if a != "average"]
        if any(set(m) - {"average"} != set(keys) for m in maps):
            raise ValueError(f"Sky {site}: azimuth profiles have different angles")
        zen_keys, zen_cos = self._sorted_by_cos(keys)
        centres = 0.5 * (self._cos_edges[1:] + self._cos_edges[:-1])
        if len(keys) != len(centres) or not np.allclose(zen_cos, centres, atol=1e-4):
            raise ValueError(f"Sky {site}: profiles must be tabulated at the bin centres")
        # Map tables in (azimuth bin, cos-zenith bin) order, the order of the weights.
        self._map_tables = [m[k] for m in maps for k in zen_keys]
        self._zen_weights = _zenith_histospline(self._cos_edges)
        self._az_weights = _histospline(self._az_edges, periodic=True)
        self._zenith_deg_arr = np.degrees(np.arccos(zen_cos))[::-1]
        self._zenith_cos_arr = zen_cos[::-1]
        self._zenith_angles = list(zen_keys)

        self._he, self._blend, self._he_tables = None, None, []
        he = description.get("high_energy")
        if he is not None and he["profile"] in library.profiles:
            self._he = library.profiles[he["profile"]]
            he_keys, self._he_cos = self._sorted_by_cos(
                [a for a in self._he if a != "average"]
            )
            self._he_tables = [self._he[k] for k in he_keys]
            self._he_bins = _linear_bin_averages(self._he_cos, self._cos_edges)
            self._blend = tuple(map(float, he["blend_GeV"]))
        elif he is not None and debug > 0:
            print(
                f"Sky {site}: high-energy profile {he['profile']} is not loaded; "
                "the map serves its own energy range only."
            )

        def common(tables):
            return set.intersection(*(set(t) for t in tables))

        map_species = common(self._map_tables)
        he_species = common(self._he_tables) if self._he_tables else set()
        joined = map_species & he_species if self._he_tables else map_species
        # Prompt leptons are not part of the map: the high-energy profile supplies
        # them at all of its energies, and conventional plus prompt is formed here.
        self._prompt = {s for s in he_species - map_species if s.startswith("pr_")}
        self._totals = {
            "total_" + s[3:]: (s[3:], s)
            for s in self._prompt
            if s[3:] in joined and "total_" + s[3:] not in joined
        }
        self._species = sorted(joined | self._prompt | set(self._totals))
        derived = dict(library.derived[description["profiles"][0]])
        derived.update(default_derived(self._species))
        self._derived = derived
        self._quantities = self._species + [
            q
            for q, (num, den) in derived.items()
            if set(num) | set(den or {}) <= set(self._species) and q not in self._species
        ]

        if self._he is not None:
            lower, upper = self._blend
            for s in joined:
                map_upper = min(t[s].domain[1] for t in self._map_tables)
                he_lower = max(t[s].domain[0] for t in self._he_tables)
                if upper > map_upper * (1 + 1e-9) or lower < he_lower * (1 - 1e-9):
                    raise ValueError(
                        f"Sky {site}: blend window {lower:g}-{upper:g} GeV must lie "
                        f"within the map (up to {map_upper:g} GeV) and the high-energy "
                        f"profile (from {he_lower:g} GeV) for {s}"
                    )

    @staticmethod
    def _sorted_by_cos(keys):
        cosines = np.cos(np.radians([float(k) for k in keys]))
        order = np.argsort(cosines)
        return [keys[i] for i in order], cosines[order]

    # ------------------------------------------------------------------ geometry
    @property
    def zenith_angles(self) -> list:
        """Zenith angles (degrees) of the map's bin centres, ascending."""
        return [format_angle(a) for a in self._zenith_deg_arr]

    @property
    def azimuth_edges(self) -> np.ndarray:
        """Azimuth bin edges in degrees (see ``azimuth_convention``)."""
        return self._az_edges.copy()

    @property
    def azimuth_convention(self) -> str:
        return self.description.get("azimuth_convention", "")

    @property
    def cos_zenith_edges(self) -> np.ndarray:
        """cos(zenith) bin edges, ascending (+1 is down-going)."""
        return self._cos_edges.copy()

    @property
    def high_energy(self):
        """``{"profile", "blend_GeV"}`` of the high-energy join, or None."""
        if self._he is None:
            return None
        return {
            "profile": self.description["high_energy"]["profile"],
            "blend_GeV": list(self._blend),
        }

    def energy_range(self, quantity):
        """(lower, upper) energy in GeV over which ``quantity`` is available.

        Quantities with prompt leptons start where the high-energy profile does.
        """
        domains = [self._domain(s) for s in self._names(quantity)]
        return max(d[0] for d in domains), min(d[1] for d in domains)

    def _domain(self, species):
        if species in self._totals:
            a, b = (self._domain(s) for s in self._totals[species])
            return max(a[0], b[0]), min(a[1], b[1])
        low = self._he_tables if species in self._prompt else self._map_tables
        high = self._he_tables if self._he is not None else self._map_tables
        return (
            max(t[species].domain[0] for t in low),
            min(t[species].domain[1] for t in high),
        )

    def _names(self, quantity):
        num, den = self._terms(quantity)
        return set(num) | set(den or {})

    def _weights(self, zenith_deg, azimuth_deg, binned):
        """Weights of the map tables (n_dir, n_azimuth * n_zenith) and of the
        high-energy angles (n_dir, n_angles) for arrays of directions."""
        cos_z = np.cos(np.radians(zenith_deg))
        n_zen, n_az = self._cos_edges.size - 1, self._az_edges.size - 1
        if binned:
            i = np.searchsorted(self._cos_edges, cos_z, side="right") - 1
            i = np.clip(i, 0, n_zen - 1)
            w_zen = np.eye(n_zen)[i]
        else:
            w_zen = self._zen_weights(cos_z)
        if azimuth_deg is None:
            w_az = np.broadcast_to(np.diff(self._az_edges) / 360.0, (cos_z.size, n_az))
        elif binned:
            x = (azimuth_deg - self._az_edges[0]) % 360.0 + self._az_edges[0]
            k = np.searchsorted(self._az_edges, x, side="right") - 1
            w_az = np.eye(n_az)[np.clip(k, 0, n_az - 1)]
        else:
            w_az = self._az_weights(azimuth_deg)
        w_map = (w_az[:, :, None] * w_zen[:, None, :]).reshape(cos_z.size, -1)
        if self._he is None:
            return w_map, None
        w_he = self._he_bins[i] if binned else _linear_weights(self._he_cos, cos_z)
        return w_map, w_he

    # ---------------------------------------------------------------- evaluation
    def _check_input(self, energy, quantity) -> None:
        assert quantity in self._quantities, (
            f"Quantity must be one of {', '.join(self._quantities)}."
        )
        lower, upper = self.energy_range(quantity)
        tolerance = 1e-12
        if np.max(energy) > upper * (1 + tolerance) or np.min(energy) < lower * (
            1 - tolerance
        ):
            hint = ""
            if self._he is None and self.description.get("high_energy"):
                hint = f"; load {self.description['high_energy']['profile']} to extend it"
            raise AssertionError(
                f"Energy out of range for {quantity}: {lower:g} - {upper:g} GeV{hint}"
            )

    def _species_flux(self, log_e, species, theta, gradient, w_map, w_he):
        """Flux (n_E, n_dir) of one species and its gradient (n_params, n_E, n_dir)."""
        if species in self._totals:
            (fc, gc), (fp, gp) = (
                self._species_flux(log_e, s, theta, gradient, w_map, w_he)
                for s in self._totals[species]
            )
            return fc + fp, (gc + gp) if gradient else None
        if species in self._prompt:
            w = np.ones(log_e.size)
        elif self._he is None:
            w = np.zeros(log_e.size)
        else:
            w = _smoothstep(log_e, *self._blend)
        f = np.zeros((log_e.size, w_map.shape[0]))
        g = np.zeros((len(self._model.names),) + f.shape) if gradient else None
        # The map below the window, the high-energy profile above, both inside.
        for mask, blend, tables, weights in (
            (w < 1.0, 1.0 - w, self._map_tables, w_map),
            (w > 0.0, w, self._he_tables, w_he),
        ):
            if not mask.any():
                continue
            used = np.flatnonzero(np.any(weights != 0.0, axis=0))
            values = [
                self._model.evaluate(tables[t][species], log_e[mask], theta, gradient)
                for t in used
            ]
            weights = weights[:, used]
            fv = np.array([v for v, _ in values])
            f[mask] += blend[mask, None] * (fv.T @ weights.T)
            if gradient:
                gv = np.array([gt for _, gt in values])
                g[:, mask] += blend[mask, None] * np.einsum("tpe,dt->ped", gv, weights)
        return f, g

    def _evaluate_directions(
        self, energy, zenith_deg, azimuth_deg, quantity, gradient, binned
    ):
        """Value (n_E, n_dir) and gradient (n_params, n_E, n_dir) of ``quantity``."""
        log_e = np.log(np.atleast_1d(np.asarray(energy, dtype=float)))
        zen, az, shape = self._directions(zenith_deg, azimuth_deg)
        w_map, w_he = self._weights(zen, az, binned)
        theta = self._theta()
        num, den = self._terms(quantity)

        def combine(weights):
            total, grad = 0.0, 0.0
            for species, c in weights.items():
                f, g = self._species_flux(log_e, species, theta, gradient, w_map, w_he)
                total = total + c * f
                if gradient:
                    grad = grad + c * g
            return total, grad

        n, gn = combine(num)
        if den is None:
            return n, gn, shape
        d, gd = combine(den)
        grad = (gn * d - n * gd) / d**2 if gradient else None
        return n / d, grad, shape

    @staticmethod
    def _directions(zenith_deg, azimuth_deg):
        if isinstance(zenith_deg, str):
            raise ValueError("A sky map takes numeric zenith angles (0-180 degrees)")
        zen = np.asarray(zenith_deg, dtype=float)
        if azimuth_deg is None:
            shape = zen.shape
            zen, az = zen.ravel(), None
        else:
            zen, az = np.broadcast_arrays(zen, np.asarray(azimuth_deg, dtype=float))
            shape = zen.shape
            zen, az = zen.ravel(), az.ravel()
        if np.any(zen < 0) or np.any(zen > 180):
            raise ValueError("Zenith angles must lie within 0-180 degrees")
        return zen, az, shape

    # -------------------------------------------------------------------- public
    def flux(
        self,
        energy,
        zenith_deg,
        quantity,
        params={},
        height_km=None,
        azimuth_deg=None,
        binned=False,
    ):
        """E^3-weighted flux in GeV^2/(cm^2 s sr) by direction.

        Parameters
        ----------
        energy : float or np.ndarray
            Energy in GeV, anywhere in ``energy_range(quantity)``: the map, the
            blend window and the high-energy profile above it.
        zenith_deg : float or np.ndarray
            Zenith angle in degrees, 0 (down-going) to 180 (up-going).
        quantity : str
            A species or derived quantity, as for other profiles.
        params : dict, optional
            Parameter shifts in units of their errors.
        azimuth_deg : float or np.ndarray, optional
            Azimuth in degrees (``azimuth_convention``), broadcast against
            ``zenith_deg``. None averages over azimuth.
        binned : bool, optional
            Return the average over the bin containing the direction instead of
            the flux in the direction.

        Returns
        -------
        np.ndarray
            Shape (n_E, *directions), squeezed.
        """
        if height_km is not None:
            raise ValueError("Sky maps have no height grid")
        self._check_input(energy, quantity)
        with self._temporary_parameters(params):
            value, _, shape = self._evaluate_directions(
                energy, zenith_deg, azimuth_deg, quantity, False, binned
            )
        return value.reshape((-1,) + shape).squeeze()

    def gradient(self, energy, zenith_deg, quantity, azimuth_deg=None, binned=False):
        """d quantity / d parameter, rows in ``params.known_parameters`` order.

        Shape (n_params, n_E, *directions); scalar directions are dropped.
        """
        self._check_input(energy, quantity)
        _, grad, shape = self._evaluate_directions(
            energy, zenith_deg, azimuth_deg, quantity, True, binned
        )
        rows = [self._index[p] for p in self._params.known_parameters]
        return grad[rows].reshape((len(rows), -1) + shape)

    def error(
        self,
        energy,
        zenith_deg,
        quantity,
        only_hadronic=False,
        height_km=None,
        azimuth_deg=None,
        binned=False,
    ):
        """1-sigma error of ``flux`` from the parameter covariance (same shape)."""
        if height_km is not None:
            raise ValueError("Sky maps have no height grid")
        if self._params.cov is None:
            raise ValueError(
                "This library has no prior covariance yet; load a calibration "
                "to evaluate uncertainties or parameter shifts"
            )
        with self._temporary_exclude_parameters("GSF" if only_hadronic else None):
            g = self.gradient(energy, zenith_deg, quantity, azimuth_deg, binned)
            error = np.sqrt(np.einsum("p...,pq,q...->...", g, self._params.cov, g))
        return error.squeeze()
