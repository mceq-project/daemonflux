# Describing the parameters of a spline library

A daemonflux spline file holds the nominal fluxes together with their
derivatives with respect to a set of model parameters: hadronic-yield knobs
and cosmic-ray (primary) flux parameters. Older files only list the parameter
names, and daemonflux guessed what each one meant from its name, e.g. anything
containing `GSF` was taken to be a primary-flux parameter. New libraries can
say this explicitly.

## The parameter description

The description is an optional dictionary stored after the covariance in the
spline file:

```python
metadata = {
    "schema_version": 1,
    "parameters": [
        {"name": "yield_a", "group": "hadronic", "units": "fractional"},
        {"name": "primary_pivot", "group": "primary", "units": "sigma"},
    ],
}
```

Each parameter says which group it belongs to (`hadronic` or `primary`) and in
which units its derivative is tabulated:

- `fractional`: the derivative is per unit relative change, e.g. per +100% of a
  yield;
- `sigma`: the derivative is per one standard deviation of its prior.

The list must follow the same order as the parameters in the file. The
`hadronic` and `primary` groups decide what `only_hadronic` uncertainties
contain, whatever the parameters are called and wherever they sit. The
description is available as `Flux.metadata`, for reference only: daemonflux
does not convert between units. As before, the `params` argument of
`Flux.flux` takes shifts in standard deviations of the covariance.

Files without a description load exactly as they always did.

## Libraries without priors

Sometimes a library is produced before its priors are settled. It can then
store `None` instead of a covariance matrix. Such a library evaluates the
nominal flux normally. Uncertainties and parameter shifts, which need a
covariance, raise an error until a calibration provides one. daemonflux never
invents a default prior.

With `uncorrelated_hadr_errors=True`, correlations between hadronic parameters
are removed. For a library with a description, each hadronic parameter keeps
its own variance. Older files keep their historical behaviour, where every
hadronic variance is set to one.

## Which calibrations fit which library

A calibration is a fit of the model parameters to muon data. The fit is done
once, on the library that contains the muon experiments. The result applies
to every library built with the same parameters: the generic US Standard
atmosphere, Kamioka, the South Pole or any other site.

daemonflux therefore checks the parameters, not the file. For a library with a
description, the first entries of the calibration's `cov_params` must be the
library's parameters, with the same names in the same order. If a calibration
entry records its position as `number`, that must agree too. Nuisance
parameters of the fit, such as detector systematics, may follow. A
`spline_sha256` entry may record which file the fit was made with; it is kept
for bookkeeping and not checked. The `params`, `cov_params` and `cov_matrix`
entries are required as before. Calibrations for older files are still matched
by parameter name.

## Other contents of the same slot

The same slot has also held height-dependent splines (`height_data`) and a
list of quantities evaluated on a linear rather than logarithmic scale
(`linear_quantities`). Files in either form keep loading. A library with a
description can carry both as additional keys of the same dictionary.
