# Response files (format version 2)

A response file tabulates, for one dataset, the nominal flux of each lepton
species and its derivatives with respect to the model parameters. It is an
HDF5 file with plain arrays, so it loads without pickle and without depending
on scipy internals. The legacy pickled spline files still load as before.

```python
from daemonflux import Flux

flux = Flux(
    spl_file=["daemonflux_bess_tev_20261001a.h5", "daemonflux_usstd_generic_20261001a.h5"],
    cal_file="daemonsplines_calibration_v1_cond_20261001a.pkl",
)
flux.flux(energies, "8.1096", "muratio", exp="bess")
```

Several files can be loaded into one `Flux` if they share a parameter basis
(the same parameter names, order and meaning). Each profile must appear in only
one file.

## Layout

```
/                       attrs: format="daemonflux-response", format_version=2
/metadata               JSON text (see below)
/covariance             optional prior covariance of the parameters
/primary_covariance     optional covariance of the primary-flux parameters
/profiles/<profile>     attr "metadata": JSON text describing the profile
/profiles/<profile>/<angle>/<species>/
    x           abscissa, increasing: GeV, or GeV/c for a muon momentum axis
    value       nominal weighted flux, positive
    jacobian    (n_params, n) relative first derivative, d ln F / d a
    curvature   optional (n_params, n) relative second derivative
```

Angles are zenith angles formatted with four decimals (`"8.1096"`), or
`"average"`. Species are `mu+`, `mu-`, `numu`, `antinumu`, `nue`, `antinue`,
optionally with the `pr_` (prompt) or `total_` (conventional plus prompt)
prefix. The weighting of `value` follows the legacy files (E³ for neutrinos,
p³-type for muons).

## Quantities

Only species are stored. The evaluator forms sums (`muflux`, `numuflux`,
`nueflux`), ratios (`muratio`, `numuratio`, `nueratio`) and `flavorratio`
from them, for each prefix whose species are present. Their uncertainties use
the exact gradient of the sum or ratio, so a ratio uncertainty includes the
correlation of its numerator and denominator. Further quantities, such as a
polarization asymmetry from helicity-split muon species, can be declared in the
metadata:

```json
"derived": [{"name": "mu+_pol",
             "numerator": {"mu+_hL": 1, "mu+_hR": -1},
             "denominator": {"mu+_hL": 1, "mu+_hR": 1}}]
```

A file may also store a quantity whose species are absent (e.g. `muflux` only).
Such a table is evaluated with the same response model as a species.

Muons may also be stored by helicity at production (MCEq): `mu+_l`, `mu+_r` and
`mu+_0` for helicity −1, +1 and 0, the same for `mu-`, with any prefix. When
all three components of a charge are present, the evaluator also forms their
sum (`mu+`, unless it is stored), the polarization
`mu+_polarization` = (N_r − N_l) / N, and, with both charges, `muflux` and
`muratio` of the sums.

## Underground tables

```
/underground/<label>    attr "metadata": JSON text (kind, site, units, derived)
/underground/<label>/<species>/
    x, value, jacobian, curvature   as for profiles; x is slant depth in
                                    km.w.e., or one entry for a total
```

The jacobian rows are the file parameters followed by the
`underground_parameters` of the file metadata (e.g. rock densities), which are
experiment nuisances outside the parameter basis. `Flux.underground(label,
quantity)` evaluates a table and `Flux.underground_gradient(label, quantity)`
its gradient. Quantities are the stored species, the sums, ratios and helicity
quantities above, and the `derived` entries of the table metadata, in the form
shown above. For example, a stopping-muon table stored by helicity declares
the charge ratio `R` and the effective polarization `P0_eff` of Super-K's
stopping muons (decay-electron efficiencies as weights):

```json
"derived": [{"name": "R",
             "numerator": {"mu+_l": 1, "mu+_r": 1, "mu+_0": 1},
             "denominator": {"mu-_l": 1, "mu-_r": 1, "mu-_0": 1}},
            {"name": "P0_eff",
             "numerator": {"mu+_l": 0.718, "mu+_r": -0.718,
                           "mu-_r": 0.051, "mu-_l": -0.051},
             "denominator": {"mu+_l": 0.718, "mu+_r": 0.718, "mu+_0": 0.718,
                             "mu-_l": 0.051, "mu-_r": 0.051, "mu-_0": 0.051}}]
```

Entries may carry further keys (`formula`, `constants`); they are kept in
`Flux.library.underground_metadata` and not used by the evaluator.

## Sky maps

A sky map gives a site's flux by direction, zenith and azimuth, for example a
3D cascade calculation with the geomagnetic cutoff. It is a description in
`/sky/<site>` that groups ordinary profiles:

```
/sky/<site>             attr "metadata": JSON text
```

```json
{"profiles": ["Kamioka_az00", "...", "Kamioka_az11"],
 "azimuth_edges_deg": [0, 30, 60, "...", 360],
 "azimuth_convention": "Honda arrival direction: 0 from South, 90 from East",
 "cos_zenith_edges": [-1.0, -0.9, "...", 1.0],
 "high_energy": {"profile": "Kamioka_1d", "blend_GeV": [40, 89]}}
```

- One profile per azimuth bin, each tabulated at the zenith angles of the
  cos(zenith) bin centres, both hemispheres. Values are bin averages.
- `high_energy` (optional) names a profile without azimuth dependence, e.g. a
  1D calculation for the same site, that takes over above the window. Inside
  the window the two are blended with a weight rising smoothly in log E
  (3t² − 2t³). The window must lie within both energy ranges. Prompt species
  (`pr_*`) come from this profile at all its energies, and `total_*` is formed
  as the map's conventional flux plus that prompt flux.
- The profiles may be stored in other files of the same basis. If the
  high-energy profile is not loaded, the map serves its own energies only.

```python
flux = Flux(bundle="kamioka.bundle.json")
sky = flux["Kamioka"]
sky.flux(E, zenith_deg=95.0, quantity="numuflux", azimuth_deg=90.0)
flux.flux(E, 95.0, "total_numuflux", exp="Kamioka", azimuth_deg=[90, 270])
sky.error(E, 95.0, "numuflux", azimuth_deg=90.0)
sky.gradient(E, 95.0, "numuflux", azimuth_deg=90.0)  # rows: sky.parameter_names
sky.covariance                                       # their covariance
```

Directions are interpolated linearly in the flux, in cos(zenith) between the
bin centres (constant beyond the outermost centres) and periodically in
azimuth. `binned=True` returns the average of the bin that contains the
direction. Without `azimuth_deg` the flux is averaged over azimuth. Zenith
angles run from 0 (down-going) to 180 degrees (up-going). The result has shape
(n_E, *directions), with `zenith_deg` and `azimuth_deg` broadcast. Each value is
a weighted sum of table entries, so `gradient` is exact. Uncertainties of
averages or ratios over several directions, such as an East/West ratio, follow
from the gradients and `covariance` with their correlations.

## Metadata

```json
{
  "format": "daemonflux-response", "format_version": 2,
  "dataset_id": "bess_tev", "build_id": "bess_tev_20261001a",
  "energy_interpolation": "linear",
  "basis_sha256": "...",
  "parameters": [
    {"name": "pi+_31G", "group": "hadronic", "units": "fractional",
     "combination": "additive", "transform": "linear", "secondary": 211},
    {"name": "p_31G", "group": "hadronic", "units": "fractional",
     "combination": "factor", "transform": "linear", "secondary": 2212},
    {"name": "GSF2026_p_1GeV", "group": "primary", "units": "sigma"}
  ]
}
```

- `group`, `units` as in [response-metadata.md](response-metadata.md).
- `combination`: `additive` (default) or `factor`, see below.
- `transform`: `linear` (default) or `log`, see below. `log` needs fractional units.
- `energy_interpolation`: `linear` (default) or `cubic`, in log abscissa and
  log value; derivatives are interpolated the same way. Tables with fewer than
  four points always use linear interpolation.
- `basis_sha256` identifies the parameter basis: the ordered `name`, `group`,
  `units`, `combination` and `transform`. The writer computes it.

Any further keys (provenance, physics settings) are kept as written and are
available in `Flux.metadata["files"]`.

## Response model

For parameter values θ, each parameter has a multiplier shift a = θ
(`linear`) or a = exp(θ) − 1 (`log`). With J and c the stored first and second
derivatives,

```
F = F0 · (1 + Σ_additive [J a + c a²/2]) · Π_factor (1 + J a + c a²/2)
```

Meson-yield and primary parameters are additive: their contributions to the
flux do not overlap. Nucleon-yield parameters act on the whole cascade below
them, so they enter as factors. With all parameters additive and linear and no
curvature, the model is that of the legacy files, F = F0 (1 + Σ J a).

A `log` transform makes a yield multiplier 1 + a = exp(θ) positive for any θ.
The flux stays linear in the multiplier; a Gaussian prior on θ is a log-normal
prior on the multiplier. Choosing such a prior is the fitter's job; daemonflux
only evaluates the parameter as declared.

Uncertainties are propagated linearly with the gradient dF/dθ at the current
parameter values: σ² = gᵀ C g. For additive linear parameters this gradient
does not depend on the values, as in the legacy files.

## Calibrations and bundles

Calibrations bind to the parameter vector by name and position, as described in
[response-metadata.md](response-metadata.md). If a calibration records
`basis_sha256`, it must equal the basis of the response files. Its
`calibration_id`, `version`, `fit_id` and `inputs` entries, when present, are
available as `Flux.calibration_info`.

A bundle manifest lists response files and a calibration with their sha256.
`Flux(bundle="bundle.json")` verifies every file before loading it:

```json
{
  "format": "daemonflux-bundle", "bundle_id": "20261001a",
  "files": [{"path": "daemonflux_bess_tev_20261001a.h5", "sha256": "..."}],
  "calibration": {"path": "daemonsplines_calibration_v1_cond_20261001a.pkl",
                  "sha256": "..."}
}
```

## Writing and converting

`daemonflux.response.write_response_file(path, metadata, profiles, ...)` writes a
file atomically and returns its sha256. `profiles[profile][angle][species]` is a
mapping with `x`, `value`, `jacobian` and optionally `curvature`.

Legacy libraries convert losslessly, because their splines are linear
interpolants of the original tables:

```
python -m daemonflux.convert daemonsplines_generic_20260923a.pkl OUTDIR \
    --group bess_tev=bess --group usstd_generic=generic --build 20260923a
```

Profiles that stored only `muflux` and `muratio` keep those tables; their
ratio uncertainties are then those of the stored ratio gradient. Files converted
from libraries without parameter metadata keep the name-based calibration
binding of those libraries.
