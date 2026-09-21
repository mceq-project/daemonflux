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

A calibration for a metadata-bearing library must contain `spline_sha256`, the
SHA-256 of the exact spline file (or a list of accepted file hashes when a fit
calibrates multiple location files). Parameter names alone do not establish
compatibility. The existing `params`, `cov_params` and `cov_matrix` entries are
still required. Legacy calibrations need no new field; when supplied, the hash
is checked for legacy files as well.
