"""Tabulated response files (format version 2) and their response model.

A response file holds, for one dataset, tables of the nominal flux of each
lepton species and its derivatives with respect to every model parameter, on
the energy (or momentum) grid of the cascade calculation. Sums and ratios such
as ``muflux`` or ``muratio`` are not stored: they are formed from the species,
which keeps ratio uncertainties exact.

Layout (HDF5)::

    /                       attrs: format="daemonflux-response", format_version=2
    /metadata               JSON text: parameters, interpolation, provenance
    /covariance             optional prior covariance of the parameters
    /primary_covariance     optional covariance of the primary-flux parameters
    /profiles/<profile>     attr "metadata": JSON text describing the profile
    /profiles/<profile>/<angle>/<species>/
        x           abscissa in GeV (or GeV/c for muon momentum), increasing
        value       nominal weighted flux, positive
        jacobian    (n_params, n) relative first derivative d ln F / d a
        curvature   optional (n_params, n) relative second derivative
    /underground/<label>    optional (e.g. "total-ug-wipp", "depth-ug-LNGS");
                            attr "metadata": JSON text (kind, units, site)
    /underground/<label>/<species>/
        x           slant depth in km.w.e. (depth curves) or one entry (totals)
        value       nominal underground muon flux or intensity, positive
        jacobian    (n_params + n_underground, n): file parameters, then the
                    underground parameters listed in metadata
        curvature   optional, same shape

Underground tables fold the surface responses through muon transport in rock
(e.g. MUTE). Their extra parameters (rock densities) are experiment nuisances:
they are listed under ``underground_parameters`` and are not part of the
parameter basis, so calibrations bind as before.

Response model, for parameter values theta and multipliers a = g(theta)
(``transform`` "linear": a = theta; "log": a = exp(theta) - 1)::

    F = F0 * (1 + sum_additive [J a + c a^2 / 2]) * prod_factor (1 + J a + c a^2 / 2)

Parameters with ``combination`` "additive" (meson yields, primary flux) enter
the sum; "factor" parameters (nucleon yields) each multiply the flux. With all
parameters additive, linear and without curvature this is the model of the
legacy spline files, F = F0 (1 + sum J a).
"""

import hashlib
import json
import os
import pathlib
import tempfile

import numpy as np

FORMAT = "daemonflux-response"
FORMAT_VERSION = 2
BUNDLE_FORMAT = "daemonflux-bundle"
INTERPOLATIONS = ("linear", "cubic")
GROUPS = ("hadronic", "primary")
UNDERGROUND_GROUPS = ("underground",)
UNITS = ("fractional", "sigma")
COMBINATIONS = ("additive", "factor")
TRANSFORMS = ("linear", "log")
SPECIES = ("mu+", "mu-", "numu", "antinumu", "nue", "antinue")
PREFIXES = ("", "pr_", "total_")
HDF5_SIGNATURE = b"\x89HDF\r\n\x1a\n"


def is_response_file(path):
    """Whether ``path`` is an HDF5 response file (by its signature)."""
    with open(path, "rb") as stream:
        return stream.read(8) == HDF5_SIGNATURE


def basis_sha256(parameters):
    """Identity of a parameter basis: ordered names and their meaning.

    A calibration fits these parameters, so it applies to every response file
    with the same basis, whatever datasets the file tabulates.
    """
    keys = ("name", "group", "units", "combination", "transform")
    basis = [[p.get(k) for k in keys] for p in parameters]
    return hashlib.sha256(json.dumps(basis).encode()).hexdigest()


def normalize_parameters(parameters, groups=GROUPS):
    """Validate parameter descriptions and fill the version-2 defaults."""
    out, names = [], set()
    for p in parameters:
        if not isinstance(p, dict) or not p.get("name"):
            raise ValueError("Each parameter needs a name")
        if p["name"] in names:
            raise ValueError(f"Duplicate parameter {p['name']}")
        names.add(p["name"])
        q = dict(p)
        q.setdefault("combination", "additive")
        q.setdefault("transform", "linear")
        if q.get("group") not in groups:
            raise ValueError(f"{q['name']}: group must be one of {groups}")
        if q.get("units") not in UNITS:
            raise ValueError(f"{q['name']}: units must be one of {UNITS}")
        if q["combination"] not in COMBINATIONS:
            raise ValueError(f"{q['name']}: combination must be one of {COMBINATIONS}")
        if q["transform"] not in TRANSFORMS:
            raise ValueError(f"{q['name']}: transform must be one of {TRANSFORMS}")
        if q["transform"] == "log" and q["units"] != "fractional":
            raise ValueError(f"{q['name']}: a log transform needs fractional units")
        out.append(q)
    if not out:
        raise ValueError("A response file needs at least one parameter")
    return out


def default_derived(species):
    """Sums and ratios formed from the stored species, for each channel prefix.

    Each entry maps a quantity to ``(numerator, denominator)``, both
    ``{species: weight}``; the denominator is None for sums.
    """
    species = set(species)
    derived = {}
    for p in PREFIXES:
        s = {k: p + k for k in SPECIES}
        pairs = (
            ("mu", "mu+", "mu-"),
            ("numu", "numu", "antinumu"),
            ("nue", "nue", "antinue"),
        )
        for label, a, b in pairs:
            if {s[a], s[b]} <= species:
                derived[f"{p}{label}flux"] = ({s[a]: 1.0, s[b]: 1.0}, None)
                derived[f"{p}{label}ratio"] = ({s[a]: 1.0}, {s[b]: 1.0})
        nu = [s[k] for k in ("numu", "antinumu", "nue", "antinue")]
        if set(nu) <= species:
            derived[f"{p}flavorratio"] = (
                {nu[0]: 1.0, nu[1]: 1.0},
                {nu[2]: 1.0, nu[3]: 1.0},
            )
    return derived


def _parse_derived(entries, species):
    """Explicit derived quantities from metadata (e.g. a polarization)."""
    derived = {}
    for entry in entries or []:
        num = {k: float(v) for k, v in entry["numerator"].items()}
        den = entry.get("denominator")
        den = None if den is None else {k: float(v) for k, v in den.items()}
        if not set(num) | set(den or {}) <= set(species):
            raise ValueError(f"Derived quantity {entry['name']} uses unknown species")
        derived[entry["name"]] = (num, den)
    return derived


class ResponseTable:
    """One tabulated species: nominal flux and derivatives on a grid."""

    def __init__(self, x, value, jacobian, curvature=None, interpolation="linear"):
        x = np.asarray(x, dtype=float)
        value = np.asarray(value, dtype=float)
        jacobian = np.atleast_2d(np.asarray(jacobian, dtype=float))
        if interpolation not in INTERPOLATIONS:
            raise ValueError(f"Unknown interpolation {interpolation}")
        if x.ndim != 1 or x.size < 1 or np.any(x <= 0) or np.any(np.diff(x) <= 0):
            raise ValueError("Abscissa must be positive and strictly increasing")
        if value.shape != x.shape or np.any(value <= 0) or not np.isfinite(value).all():
            raise ValueError("Nominal values must be positive and finite on the grid")
        if jacobian.shape[1] != x.size or not np.isfinite(jacobian).all():
            raise ValueError("Jacobian must be finite with one column per grid point")
        if curvature is not None:
            curvature = np.asarray(curvature, dtype=float)
            if curvature.shape != jacobian.shape or not np.isfinite(curvature).all():
                raise ValueError("Curvature must match the jacobian")
        self.x, self.value, self.jacobian, self.curvature = x, value, jacobian, curvature
        self.n_params = jacobian.shape[0]
        self.log_x = np.log(x)
        self.interpolation = interpolation if x.size >= 4 else "linear"
        rows = [np.log(value)[None, :], jacobian]
        if curvature is not None:
            rows.append(curvature)
        self._stack = np.vstack(rows)
        self._spline = None
        if self.interpolation == "cubic":
            from scipy.interpolate import CubicSpline

            self._spline = CubicSpline(self.log_x, self._stack, axis=1)

    @property
    def domain(self):
        return self.x[0], self.x[-1]

    def interpolate(self, log_e):
        """Return (F0, J, C) at ``log_e``; C is None without curvature."""
        log_e = np.atleast_1d(np.asarray(log_e, dtype=float))
        if self.x.size == 1:  # a single value (e.g. an underground total)
            stack = np.repeat(self._stack, log_e.size, axis=1)
        elif self._spline is not None:
            stack = self._spline(log_e)
        else:
            i = np.clip(np.searchsorted(self.log_x, log_e) - 1, 0, self.x.size - 2)
            w = (log_e - self.log_x[i]) / (self.log_x[i + 1] - self.log_x[i])
            stack = self._stack[:, i] * (1 - w) + self._stack[:, i + 1] * w
        n = self.n_params
        curvature = stack[1 + n :] if self.curvature is not None else None
        return np.exp(stack[0]), stack[1 : 1 + n], curvature


class ResponseModel:
    """Combine per-parameter responses into a flux and its parameter gradient."""

    def __init__(self, parameters, groups=GROUPS):
        self.parameters = normalize_parameters(parameters, groups)
        self.names = [p["name"] for p in self.parameters]
        self.factor = np.array([p["combination"] == "factor" for p in self.parameters])
        self.log = np.array([p["transform"] == "log" for p in self.parameters])

    def multipliers(self, theta):
        """Multiplier shifts a(theta) and their derivatives da/dtheta."""
        theta = np.asarray(theta, dtype=float)
        a = np.where(self.log, np.expm1(theta), theta)
        da = np.where(self.log, np.exp(theta), 1.0)
        return a, da

    def evaluate(self, table, log_e, theta, gradient=False):
        """Flux F and, optionally, dF/dtheta with shape (n_params, n_energies)."""
        f0, jac, curv = table.interpolate(log_e)
        a, da = self.multipliers(theta)
        col = a[:, None]
        terms = jac * col
        slopes = jac.copy()
        if curv is not None:
            terms = terms + 0.5 * curv * col**2
            slopes = slopes + curv * col
        add, fac = ~self.factor, self.factor
        s_add = 1.0 + terms[add].sum(axis=0)
        factors = 1.0 + terms[fac]
        s_fac = np.prod(factors, axis=0) if fac.any() else np.ones_like(f0)
        flux = f0 * s_add * s_fac
        if not gradient:
            return flux, None
        grad = np.empty_like(jac)
        grad[add] = f0 * s_fac * slopes[add] * da[add, None]
        if fac.any():
            # Product of all other factors, without dividing by a factor.
            ones = np.ones_like(f0)[None, :]
            before = np.cumprod(np.vstack([ones, factors[:-1]]), axis=0)
            after = np.cumprod(np.vstack([ones, factors[::-1][:-1]]), axis=0)[::-1]
            others = before * after
            grad[fac] = f0 * s_add * others * slopes[fac] * da[fac, None]
        return flux, grad


class ResponseLibrary:
    """Contents of one or more response files that share a parameter basis."""

    def __init__(self):
        self.parameters = None
        self.metadata = []
        self.covariance = None
        self.primary_covariance = None
        self.profiles = {}
        self.profile_metadata = {}
        self.derived = {}
        self.underground_parameters = []
        self.underground = {}
        self.underground_metadata = {}
        self.files = []

    @property
    def names(self):
        return [p["name"] for p in self.parameters]

    @property
    def basis_sha256(self):
        return basis_sha256(self.parameters)

    def add(self, path, expected_sha256=None):
        """Read one response file; its basis must match those already read."""
        import h5py

        path = pathlib.Path(path)
        sha = file_sha256(path)
        if expected_sha256 is not None and sha != expected_sha256:
            raise ValueError(
                f"{path.name}: sha256 {sha} does not match {expected_sha256}"
            )
        with h5py.File(path, "r") as h5:
            if h5.attrs.get("format") != FORMAT:
                raise ValueError(f"{path.name} is not a daemonflux response file")
            if int(h5.attrs.get("format_version", -1)) != FORMAT_VERSION:
                raise ValueError(f"{path.name}: unsupported format version")
            meta = json.loads(h5["metadata"].asstr()[()])
            params = normalize_parameters(meta["parameters"])
            if self.parameters is None:
                self.parameters = params
            elif basis_sha256(params) != self.basis_sha256:
                raise ValueError(f"{path.name} has a different parameter basis")
            for key, attr in (
                ("covariance", "covariance"),
                ("primary_covariance", "primary_covariance"),
            ):
                if key in h5:
                    matrix = np.asarray(h5[key][()], dtype=float)
                    current = getattr(self, attr)
                    if current is not None and not np.allclose(
                        current, matrix, rtol=1e-10, atol=0
                    ):
                        raise ValueError(f"{path.name}: {key} differs from earlier files")
                    setattr(self, attr, matrix)
            interpolation = meta.get("energy_interpolation", "linear")
            for profile, group in h5["profiles"].items():
                if profile in self.profiles:
                    raise ValueError(f"Profile {profile} appears in more than one file")
                text = group.attrs.get("metadata", "{}")
                self.profile_metadata[profile] = json.loads(text)
                angles = {}
                for angle, species_group in group.items():
                    tables = {}
                    for species, d in species_group.items():
                        tables[species] = ResponseTable(
                            d["x"][()],
                            d["value"][()],
                            d["jacobian"][()],
                            d["curvature"][()] if "curvature" in d else None,
                            interpolation,
                        )
                        if tables[species].n_params != len(self.parameters):
                            raise ValueError(
                                f"{profile}/{angle}/{species}: wrong jacobian rows"
                            )
                    angles[angle] = tables
                self.profiles[profile] = angles
                species = {s for t in angles.values() for s in t}
                derived = default_derived(species)
                derived.update(_parse_derived(meta.get("derived"), species))
                self.derived[profile] = derived
            self._add_underground(h5, meta, path.name, interpolation)
        meta["file"] = path.name
        meta["sha256"] = sha
        self.metadata.append(meta)
        self.files.append({"path": str(path), "sha256": sha})
        return self


    def _add_underground(self, h5, meta, name, interpolation):
        if "underground" not in h5:
            return
        extra = normalize_parameters(
            meta.get("underground_parameters", []), UNDERGROUND_GROUPS
        ) if meta.get("underground_parameters") else []
        if self.underground and extra != self.underground_parameters:
            raise ValueError(f"{name}: underground parameters differ from earlier files")
        self.underground_parameters = extra
        rows = len(self.parameters) + len(extra)
        for label, group in h5["underground"].items():
            if label in self.underground:
                raise ValueError(f"Underground table {label} appears in more than one file")
            self.underground_metadata[label] = json.loads(group.attrs.get("metadata", "{}"))
            tables = {}
            for species, d in group.items():
                tables[species] = ResponseTable(
                    d["x"][()],
                    d["value"][()],
                    d["jacobian"][()],
                    d["curvature"][()] if "curvature" in d else None,
                    interpolation,
                )
                if tables[species].n_params != rows:
                    raise ValueError(f"underground/{label}/{species}: wrong jacobian rows")
            self.underground[label] = tables


def read_response_files(paths, expected=None):
    """Load response files into one library. ``expected`` maps path -> sha256."""
    library = ResponseLibrary()
    for path in paths:
        library.add(path, (expected or {}).get(str(path)))
    return library


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_response_file(
    path,
    metadata,
    profiles,
    covariance=None,
    primary_covariance=None,
    profile_metadata=None,
    underground=None,
    underground_metadata=None,
):
    """Write a version-2 response file atomically and return its sha256.

    ``profiles[profile][angle][species]`` is a mapping with ``x``, ``value``,
    ``jacobian`` and optionally ``curvature``. ``metadata`` must contain
    ``parameters``; ``energy_interpolation`` defaults to "linear". Species
    tables are validated with the same checks the reader applies.

    ``underground[label][species]`` holds underground tables whose jacobians
    have one row per file parameter followed by one per entry of
    ``metadata["underground_parameters"]``.
    """
    import h5py

    meta = dict(metadata)
    meta["parameters"] = normalize_parameters(meta["parameters"])
    meta.setdefault("energy_interpolation", "linear")
    if meta["energy_interpolation"] not in INTERPOLATIONS:
        raise ValueError("energy_interpolation must be linear or cubic")
    meta["basis_sha256"] = basis_sha256(meta["parameters"])
    meta["format"], meta["format_version"] = FORMAT, FORMAT_VERSION
    n = len(meta["parameters"])
    if meta.get("underground_parameters"):
        meta["underground_parameters"] = normalize_parameters(
            meta["underground_parameters"], UNDERGROUND_GROUPS
        )
    n_underground = n + len(meta.get("underground_parameters", []))
    if not profiles and not underground:
        raise ValueError("A response file needs at least one profile")
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".part")
    os.close(fd)
    try:
        with h5py.File(tmp, "w") as h5:
            h5.attrs["format"], h5.attrs["format_version"] = FORMAT, FORMAT_VERSION
            h5.create_dataset("metadata", data=json.dumps(meta, default=_json_default))
            for key, matrix in (
                ("covariance", covariance),
                ("primary_covariance", primary_covariance),
            ):
                if matrix is not None:
                    h5.create_dataset(key, data=np.asarray(matrix, dtype=float))
            root = h5.create_group("profiles")
            for profile, angles in (profiles or {}).items():
                group = root.create_group(profile)
                group.attrs["metadata"] = json.dumps(
                    (profile_metadata or {}).get(profile, {}), default=_json_default
                )
                for angle, species in angles.items():
                    angle_group = group.create_group(angle)
                    for name, table in species.items():
                        checked = ResponseTable(
                            table["x"],
                            table["value"],
                            table["jacobian"],
                            table.get("curvature"),
                        )
                        if checked.n_params != n:
                            raise ValueError(
                                f"{profile}/{angle}/{name}: {checked.n_params} rows, {n} parameters"
                            )
                        d = angle_group.create_group(name)
                        opts = {"compression": "gzip", "shuffle": True}
                        d.create_dataset("x", data=checked.x)
                        d.create_dataset("value", data=checked.value)
                        d.create_dataset("jacobian", data=checked.jacobian, **opts)
                        if checked.curvature is not None:
                            d.create_dataset("curvature", data=checked.curvature, **opts)
            if underground:
                root = h5.create_group("underground")
            for label, species in (underground or {}).items():
                if "/" in label:
                    raise ValueError(f"Underground label {label} must not contain '/'")
                group = root.create_group(label)
                group.attrs["metadata"] = json.dumps(
                    (underground_metadata or {}).get(label, {}), default=_json_default
                )
                for name, table in species.items():
                    checked = ResponseTable(
                        table["x"], table["value"], table["jacobian"], table.get("curvature")
                    )
                    if checked.n_params != n_underground:
                        raise ValueError(
                            f"underground/{label}/{name}: {checked.n_params} rows, "
                            f"{n_underground} parameters"
                        )
                    d = group.create_group(name)
                    d.create_dataset("x", data=checked.x)
                    d.create_dataset("value", data=checked.value)
                    d.create_dataset("jacobian", data=checked.jacobian)
                    if checked.curvature is not None:
                        d.create_dataset("curvature", data=checked.curvature)
        umask = os.umask(0)
        os.umask(umask)
        os.chmod(tmp, 0o666 & ~umask)
        os.replace(tmp, path)
    finally:
        pathlib.Path(tmp).unlink(missing_ok=True)
    return file_sha256(path)


def _json_default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    raise TypeError(f"Cannot serialize {type(obj).__name__}")


def read_bundle(path):
    """Read a bundle manifest: response files and calibration with their sha256.

    Relative paths are resolved against the manifest's directory.
    """
    path = pathlib.Path(path)
    bundle = json.loads(path.read_text())
    if bundle.get("format") != BUNDLE_FORMAT:
        raise ValueError(f"{path.name} is not a daemonflux bundle manifest")
    base = path.parent

    def resolve(entry):
        p = pathlib.Path(entry["path"])
        return str(p if p.is_absolute() else base / p), entry.get("sha256")

    files = [resolve(e) for e in bundle["files"]]
    calibration = resolve(bundle["calibration"]) if bundle.get("calibration") else None
    return bundle, files, calibration
