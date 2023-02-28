# daemonflux: DAta-drivEn and MuOn-calibrated Neutrino flux

Daemonflux is a tabulated/splined version of the an atmospheric flux model calibrated on muon spectrometer data.

## Requirements
 * `Python > 3.7`, `numpy`, `scipy`
 * `matplotlib` for examples

## Installation
a) From PyPi: 
    
    pip install daemonflux
    
b) From source in editable mode, so the package gets updated after each `git pull`:
```bash
$ git clone https://github.com/mceq-project/daemonflux
$ cd daemonflux
$ python3 -m pip install -e .
```

## Usage

Follow the [example](examples/example.ipynb), where more features are demonstrated. But in a nutshell, calculating calibrated fluxes from the provided tables works like:

    from daemonflux import Flux
    import numpy as np
    import matplotlib.pyplot as plt

    daemonflux = Flux(location='generic')
    egrid = np.logspace(0,5) # Energy in GeV

    fl = daemonflux.flux(egrid, '15', 'numuflux')
    err = daemonflux.error(egrid, '15', 'numuflux')
    plt.loglog(egrid, fl, color='k')
    plt.fill_between(egrid, fl + err, fl - err,
        color='r', alpha=.3, label=r'1$\sigma$ error')
    ...

Resulting in the following figure:

![Muon Neutrino Flux plot](flux_example.png "Muon neutrino flux scaled by $E^3$ for clarity.")

## Citation

Coming soon.

## LICENSE

[BSD 3-Clause License](LICENSE)
