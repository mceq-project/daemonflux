"""Sky maps: fluxes by zenith and azimuth, joined to a 1D table at high energy.

A sky map (``/sky/<site>`` in a response file, see ``response.check_sky``)
groups one profile per azimuth bin, each tabulated at the zenith-bin centres
with bin-averaged values. Above an energy window the map hands over to a
profile without azimuth dependence, typically an ordinary 1D cascade
calculation for the same site, where geomagnetic and 3D effects have vanished.
Only the tables are stored; the join happens here, at evaluation time.

Directions are interpolated linearly in the flux: in cos(zenith) between the
bin centres (constant beyond the outermost centres) and periodically in
azimuth between the bin centres. ``binned=True`` returns the bin average of the
bin containing the direction instead. Every value is a weighted sum of table
entries, so gradients, errors and ratios between directions stay exact.
"""

import numpy as np

from .flux import _ResponseEntry
from .response import default_derived
from .utils import format_angle


def _smoothstep(log_e, lower, upper):
    """Weight of the high-energy profile: 0 below ``lower``, 1 above ``upper``."""
    t = np.clip((log_e - np.log(lower)) / (np.log(upper) - np.log(lower)), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _linear_weights(nodes, x):
    """[(index, weight)] interpolating ascending ``nodes`` at x, constant outside."""
    if x <= nodes[0]:
        return [(0, 1.0)]
    if x >= nodes[-1]:
        return [(len(nodes) - 1, 1.0)]
    j = int(np.searchsorted(nodes, x))
    t = (x - nodes[j - 1]) / (nodes[j] - nodes[j - 1])
    return [(j - 1, 1.0 - t), (j, t)]


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
        self._maps = [library.profiles[p] for p in description["profiles"]]
        self._az_edges = np.asarray(description["azimuth_edges_deg"], dtype=float)
        self._az_centres = 0.5 * (self._az_edges[1:] + self._az_edges[:-1])
        self._cos_edges = np.asarray(description["cos_zenith_edges"], dtype=float)

        keys = [a for a in self._maps[0] if a != "average"]
        if any(set(m) - {"average"} != set(keys) for m in self._maps):
            raise ValueError(f"Sky {site}: azimuth profiles have different angles")
        self._zen_keys, self._zen_cos = self._sorted_by_cos(keys)
        centres = 0.5 * (self._cos_edges[1:] + self._cos_edges[:-1])
        if len(keys) != len(centres) or not np.allclose(
            self._zen_cos, centres, atol=1e-4
        ):
            raise ValueError(f"Sky {site}: profiles must be tabulated at the bin centres")

        self._he, self._blend = None, None
        tables = [t for m in self._maps for a, t in m.items() if a != "average"]
        he = description.get("high_energy")
        if he is not None and he["profile"] in library.profiles:
            self._he = library.profiles[he["profile"]]
            self._he_keys, self._he_cos = self._sorted_by_cos(
                [a for a in self._he if a != "average"]
            )
            self._blend = tuple(map(float, he["blend_GeV"]))
        elif he is not None and debug > 0:
            print(
                f"Sky {site}: high-energy profile {he['profile']} is not loaded; "
                "the map serves its own energy range only."
            )
        he_tables = [t for a, t in (self._he or {}).items() if a != "average"]
        map_species = set.intersection(*(set(t) for t in tables))
        he_species = (
            set.intersection(*(set(t) for t in he_tables)) if he_tables else set()
        )
        joined = map_species & he_species if he_tables else map_species
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
        self._index = {name: i for i, name in enumerate(model.names)}
        # Attributes of the base classes that do not apply to a sky map.
        self._tables, self._fl_spl, self._jac_spl = {}, {}, {}
        self._spl_contains_average = False
        self._zenith_angles = list(self._zen_keys)
        self._zenith_deg_arr = np.degrees(np.arccos(self._zen_cos))[::-1]
        self._zenith_cos_arr = self._zen_cos[::-1]

        if self._he is not None:
            lower, upper = self._blend
            for s in joined:
                map_upper = min(t[s].domain[1] for t in tables)
                he_lower = max(t[s].domain[0] for t in he_tables)
                if upper > map_upper * (1 + 1e-9) or lower < he_lower * (1 - 1e-9):
                    raise ValueError(
                        f"Sky {site}: blend window {lower:g}-{upper:g} GeV must lie "
                        f"within the map (up to {map_upper:g} GeV) and the high-energy "
                        f"profile (from {he_lower:g} GeV) for {s}"
                    )
        self._map_tables, self._he_tables = tables, he_tables

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

    def _zenith_bin(self, cos_z):
        i = int(np.searchsorted(self._cos_edges, cos_z, side="right")) - 1
        return min(max(i, 0), len(self._cos_edges) - 2)

    def _map_weights(self, zenith_deg, azimuth_deg, binned):
        """[(azimuth bin, zenith key, weight)] of the map tables for one direction."""
        cos_z = np.cos(np.radians(zenith_deg))
        if binned:
            zen = [(self._zenith_bin(cos_z), 1.0)]
        else:
            zen = _linear_weights(self._zen_cos, cos_z)
        n = len(self._az_centres)
        if azimuth_deg is None:
            width = np.diff(self._az_edges) / 360.0
            az = [(k, width[k]) for k in range(n)]
        elif binned:
            x = (azimuth_deg - self._az_edges[0]) % 360.0 + self._az_edges[0]
            k = int(np.searchsorted(self._az_edges, x, side="right")) - 1
            az = [(min(k, n - 1), 1.0)]
        else:
            offsets = np.append(self._az_centres - self._az_centres[0], 360.0)
            x = (azimuth_deg - self._az_centres[0]) % 360.0
            k = int(np.searchsorted(offsets, x, side="right")) - 1
            t = (x - offsets[k]) / (offsets[k + 1] - offsets[k])
            az = [(k, 1.0 - t), ((k + 1) % n, t)]
        return [(k, self._zen_keys[i], wa * wz) for k, wa in az for i, wz in zen]

    def _he_weights(self, zenith_deg, binned):
        cos_z = np.cos(np.radians(zenith_deg))
        if binned:
            i = self._zenith_bin(cos_z)
            cos_z = 0.5 * (self._cos_edges[i] + self._cos_edges[i + 1])
        return [(self._he_keys[i], w) for i, w in _linear_weights(self._he_cos, cos_z)]

    # ---------------------------------------------------------------- evaluation
    def _check_input(self, energy, quantity) -> None:
        assert (
            quantity in self._quantities
        ), f"Quantity must be one of {', '.join(self._quantities)}."
        lower, upper = self.energy_range(quantity)
        tolerance = 1e-12
        if np.max(energy) > upper * (1 + tolerance) or np.min(energy) < lower * (
            1 - tolerance
        ):
            hint = ""
            if self._he is None and self.description.get("high_energy"):
                hint = (
                    f"; load {self.description['high_energy']['profile']} " "to extend it"
                )
            raise AssertionError(
                f"Energy out of range for {quantity}: {lower:g} - {upper:g} GeV{hint}"
            )

    def _species_direction(
        self, log_e, zenith_deg, azimuth_deg, species, theta, gradient, binned
    ):
        """Flux of one species in one direction and its gradient over all parameters."""
        if species in self._totals:
            (fc, gc), (fp, gp) = (
                self._species_direction(
                    log_e, zenith_deg, azimuth_deg, s, theta, gradient, binned
                )
                for s in self._totals[species]
            )
            return fc + fp, (gc + gp) if gradient else None
        f = np.zeros(log_e.size)
        g = np.zeros((len(self._model.names), log_e.size)) if gradient else None
        if species in self._prompt:
            w_he = np.ones(log_e.size)
        elif self._he is None:
            w_he = np.zeros(log_e.size)
        else:
            w_he = _smoothstep(log_e, *self._blend)
        parts = []
        low = w_he < 1.0
        if low.any():
            parts.append(
                (
                    low,
                    1.0 - w_he[low],
                    [
                        (self._maps[k][key][species], w)
                        for k, key, w in self._map_weights(
                            zenith_deg, azimuth_deg, binned
                        )
                    ],
                )
            )
        high = w_he > 0.0
        if high.any():
            parts.append(
                (
                    high,
                    w_he[high],
                    [
                        (self._he[key][species], w)
                        for key, w in self._he_weights(zenith_deg, binned)
                    ],
                )
            )
        for mask, blend, terms in parts:
            for table, w in terms:
                fv, gv = self._model.evaluate(table, log_e[mask], theta, gradient)
                f[mask] += w * blend * fv
                if gradient:
                    g[:, mask] += w * blend * gv
        return f, g

    def _evaluate_directions(
        self, energy, zenith_deg, azimuth_deg, quantity, gradient, binned
    ):
        """Value (n_E, n_dir) and gradient (n_params, n_E, n_dir) of ``quantity``."""
        log_e = np.log(np.atleast_1d(np.asarray(energy, dtype=float)))
        zen, az, shape = self._directions(zenith_deg, azimuth_deg)
        theta = self._theta()
        num, den = self._terms(quantity)
        n_e, n_p = log_e.size, len(self._model.names)
        value = np.empty((n_e, zen.size))
        grad = np.empty((n_p, n_e, zen.size)) if gradient else None
        for d in range(zen.size):
            a = None if az is None else az[d]

            def combine(weights):
                total, tgrad = 0.0, 0.0
                for species, w in weights.items():
                    f, g = self._species_direction(
                        log_e, zen[d], a, species, theta, gradient, binned
                    )
                    total = total + w * f
                    if gradient:
                        tgrad = tgrad + w * g
                return total, tgrad

            n, gn = combine(num)
            if den is None:
                value[:, d] = n
                if gradient:
                    grad[:, :, d] = gn
                continue
            dd, gd = combine(den)
            value[:, d] = n / dd
            if gradient:
                grad[:, :, d] = (gn * dd - n * gd) / dd**2
        return value, grad, shape

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
            Return the average of the bin containing the direction instead of
            interpolating between bin centres.

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
