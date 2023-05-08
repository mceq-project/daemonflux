import sys
from daemonflux.flux import Flux
from importlib.metadata import version
from scipy.version import version as scipy_version
import scipy.interpolate  # noqa: F401

__version__ = version("daemonflux")
__all__ = ["Flux", "__version__"]

if int(scipy_version.split(".")[1]) < 8:
    sys.modules["scipy.interpolate._fitpack2"] = sys.modules[
        "scipy.interpolate.fitpack2"
    ]
    raise DeprecationWarning(
        "scipy version must be >= 1.8.0. "
        + "Support for lower version will be removed in future releases."
    )
