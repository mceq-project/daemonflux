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
