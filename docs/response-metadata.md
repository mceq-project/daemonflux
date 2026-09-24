# Response-library metadata

Legacy four- and five-item spline payloads remain supported. A new library can
append a sixth dictionary:

```python
metadata = {
    "schema_version": 1,
    "parameters": [
        {"name": "yield_a", "group": "hadronic", "units": "fractional"},
        {"name": "primary_pivot", "group": "primary", "units": "sigma"},
    ],
}
```

The parameter order must exactly match the payload labels. Groups determine
hadronic-only errors, independently of naming and position. Covariance and
calibrated values must use the units of the stored Jacobian. This metadata is
available as `Flux.metadata`; it does not convert parameter units. The public
`params` argument continues to express shifts in covariance-derived standard
deviations.

A library whose priors have not been chosen may store `None` as its covariance.
It supports nominal flux evaluation, but uncertainty evaluation and sigma-based
parameter shifts raise an error until a calibration supplies a covariance.
No unit prior is inferred. For metadata-bearing files, the uncorrelated-hadronic
option preserves the supplied variances while removing within-hadronic
correlations. Legacy files retain their historical unit-diagonal behavior.

A calibration is a fit of the parameter vector to the muon data, carried out on
the private experiment library. It is distributed with any downstream library
that shares that parameter vector: `generic`/USStd, Kamioka, South Pole or other
custom sites. For a metadata-bearing library, the first entries of `cov_params`
must therefore equal the library's parameter names in order. A stored `number`
in `params` must equal the parameter's position. Nuisance parameters may follow
the physics block. `spline_sha256` may record the library the fit was made
against; it is provenance only and is not checked on load. The existing
`params`, `cov_params` and `cov_matrix` entries are still required. Legacy
libraries keep name-based reordering of the calibration covariance.
