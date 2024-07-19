from typing import Dict, Union, Tuple, Generator, List
import numpy as np
import pickle
import pathlib
from .utils import grid_cov, is_iterable, format_angle, _cached_data_dir
from contextlib import contextmanager

# # Anatoli's installation requires me to add this
# import sys
# from scipy.interpolate import fitpack2
# sys.modules["scipy.interpolate._fitpack2"] = sys.modules["scipy.interpolate.fitpack2"]


base_path = pathlib.Path(__file__).parent.absolute()


class Parameters:
    """Class to store parameters and their covariance matrix.

    Parameters
    ----------
    known_parameters : list of str
        List of parameter names.
    values : np.ndarray, shape (n_params,)
        Array of parameter values.
    cov : np.ndarray, shape (n_params, n_params)
        Covariance matrix of parameters.
    """

    def __init__(
        self, known_parameters: List[str], values: np.ndarray, cov: np.ndarray
    ):
        self.known_parameters = known_parameters
        self._n_non_gsf = len([p for p in known_parameters if "GSF" not in p])
        self.values = values
        self._unmodified_values = np.copy(values)
        self.cov = cov

    @property
    def invcov(self) -> np.ndarray:
        """Inverse of the covariance matrix of parameters.

        Returns
        -------
        np.ndarray, shape (n_params, n_params)
            Inverse of the covariance matrix.
        """
        return np.linalg.inv(self.cov)

    @property
    def errors(self) -> np.ndarray:
        """Errors of parameters.

        Returns
        -------
        np.ndarray, shape (n_params,)
            Array of parameter errors.
        """
        return np.sqrt(np.diag(self.cov))

    @property
    def chi2(self) -> float:
        """
        Returns the chi-square value of the parameters.

        Parameters
        ----------
        params: dict
            Dictionary of modified parameters

        Returns
        -------
        numpy.ndarray
            Chi-square value associated with params
        """
        return np.sum(grid_cov(self.values - self._unmodified_values, self.invcov))

    def __iter__(self) -> Generator[Tuple[str, float], None, None]:
        """Iterate over the parameters.

        Yields
        ------
        str
            Parameter name.
        float
            Parameter value.
        """
        for p, pv in zip(self.known_parameters, self.values):
            yield p, pv


class Flux:
    """
    This class encapsulates the behavior of a Flux.

    Attributes
    ----------
    exclude : list
        A list of items to be excluded.
    _debug : int
        Debug level.
    supported_fluxes : list
        A list of fluxes supported.

    """

    _default_url = (
        "https://github.com/mceq-project/daemonflux/releases/download/prerelease/"
    )
    _default_spl_file = "daemonsplines_{location}_{rev}.pkl"
    _default_cal_file = "daemonsplines_calibration_{cset}_{rev}.pkl"
    _revision = "202303_2"

    def __init__(
        self,
        location="generic",
        spl_file=None,
        cal_file=None,
        calibration_set="default",
        use_calibration=True,
        uncorrelated_hadr_errors=False,
        exclude=[],
        keep_old_revisions=False,
        debug=1,
    ) -> None:
        """
        Initialize the Flux instance.

        Parameters
        ----------
        location : str, optional
            Location to be used, default is "generic".
        spl_file : str, optional
            Path to the spline file.
        cal_file : str, optional
            Path to the calibration file.
        use_calibration : bool, optional
            Flag indicating whether to use calibration, default is True.
        uncorrelated_hadr_errors: bool, optional
            Flag indicating whether to use uncorrelated hadronic errors
            (during calibration), default is False.
        calibration_set : str, optional
            Calibration set to be used. Default is "default". Optional is "with_deis".
        exclude : list, optional
            A list of parameters to be excluded, default is an empty list.
        keep_old_revisions : bool, optional
            Flag indicating whether to keep old spline file revisions, default is False.
        debug : int, optional
            Debug level, default is 1.
        """
        self.exclude = exclude
        self._debug = debug
        self._uncorrelated_hadr_errors = uncorrelated_hadr_errors

        # Define location or spl_file
        assert location or spl_file, "Either location or spl_file must be defined."

        spl_file = (
            spl_file
            if spl_file
            else _cached_data_dir(
                self._default_url
                + self._default_spl_file.format(location=location, rev=self._revision)
            )
        )
        if not use_calibration or not calibration_set:
            cal_file = None
        elif use_calibration and cal_file is None:
            cal_file = _cached_data_dir(
                self._default_url
                + self._default_cal_file.format(
                    cset=calibration_set, rev=self._revision
                )
            )

        self._load_splines(spl_file, cal_file)
        if not keep_old_revisions:
            self._cleanup_old_revisions()

    def _cleanup_old_revisions(self):
        """
        Clean up old revisions of the data.
        """
        import pathlib

        for f in pathlib.Path(base_path / "data").glob("daemonsplines*"):
            if self._revision not in str(f):
                print("Removing old version", f)
                f.unlink()

    def _load_splines(self, spl_file, cal_file):
        """
        Load splines from given files.

        Parameters
        ----------
        spl_file : str
            Path to the spline file.
        cal_file : str
            Path to the calibration file.
        """
        from .utils import rearrange_covariance
        from copy import deepcopy

        assert pathlib.Path(spl_file).is_file(), f"Spline file {spl_file} not found."
        with open(spl_file, "rb") as f:
            if self._debug > 2:
                print("Loading splines from", spl_file)
            (
                known_pars,
                self._fl_spl,
                self._jac_spl,
                cov,
            ) = pickle.load(
                f
            )[:4]

        known_parameters = []
        for k in known_pars:
            if k in self.exclude:
                continue
            if k.startswith("total_"):
                continue
            known_parameters.append(k)

        if cal_file is None:
            print("No calibration used.")
            params = Parameters(
                known_parameters,
                np.zeros(len(known_parameters)),
                cov,
            )
            assert params.cov.shape == (len(known_parameters),) * 2, (
                f"covariance shape {params.cov.shape} is not consistent"
                + f" with the number of parameters {len(known_parameters)}"
            )
        else:
            assert pathlib.Path(
                cal_file
            ).is_file(), f"Calibration file {cal_file} not found."
            with open(str(cal_file), "rb") as f:
                if self._debug > 2:
                    print("Loading calibration from", cal_file)
                calibration_d = pickle.load(f, encoding="latin1")

            param_values = []
            for ip, n in enumerate(known_parameters):
                try:
                    param_values.append(calibration_d["params"][n]["value"])
                except KeyError:
                    raise KeyError("No calibration for", n)

            # Reorder the covariance such that it corresponds to the know_params order
            original_param_order = dict(
                [
                    (p, ip)
                    for ip, p in enumerate(calibration_d["cov_params"])
                    if p in known_parameters
                ]
            )
            n_physics_params = max(original_param_order.values()) + 1

            assert sorted(original_param_order.keys()) == sorted(
                known_parameters
            ), "Parameters inconsistent between spl and calibration file"

            # Create a new covariance with the correct order of parameters
            cov = rearrange_covariance(
                original_param_order,
                known_parameters,
                calibration_d["cov_matrix"][:n_physics_params, :n_physics_params],
            )

            # Check if the the rearranged cov is correct
            for ip, pi in enumerate(known_parameters):
                for jp, pj in enumerate(known_parameters):
                    assert (
                        calibration_d["cov_matrix"][
                            original_param_order[pi], original_param_order[pj]
                        ]
                        == cov[ip, jp]
                    ), (
                        "Covariance for parameters "
                        + str(pi)
                        + " "
                        + str(pj)
                        + " incorrectly sorted."
                    )

            params = Parameters(known_parameters, np.asarray(param_values), cov)

        if self._uncorrelated_hadr_errors:
            params.cov[: params._n_non_gsf, : params._n_non_gsf] = np.diag(
                np.ones(params._n_non_gsf)
            )

        # If multiple locations inside the spline file, create a FluxEntry for each
        self.supported_fluxes = []
        for exp in self._fl_spl:
            subflux = _FluxEntry(
                exp,
                self._fl_spl[exp],
                self._jac_spl[exp],
                deepcopy(params),
                self._debug,
            )
            setattr(self, exp, subflux)
            self.supported_fluxes.append(exp)

    def __repr__(self):
        s = ""
        for exp in self._fl_spl:
            s += "{0}: [{1}]\n".format(exp, ", ".join(self._fl_spl[exp].keys()))
        return s

    def print_locations(self):
        """
        Print the locations contained within the spline file.
        """
        print(self.__repr__())

    @property
    def zenith_angles(self, exp=""):
        """
        Retrieve zenith angles of splines for the specified location/experiment.

        Parameters
        ----------
        exp : str, optional
            Experiment name, default is an empty string.
        """
        if not exp and len(self.supported_fluxes) > 1:
            raise Exception("'exp' argument needs to be one of", self.supported_fluxes)
        if len(self.supported_fluxes) == 1:
            return self.__getattribute__(self.supported_fluxes[0]).zenith_angles
        else:
            return self.__getattribute__(exp).zenith_angles

    def _get_flux_instance(self, exp):
        """
        Retrieve the _FluxEntry indexed by an experiment or location.

        Parameters
        ----------
        exp : str
            Experiment name.
        """
        if not exp and len(self.supported_fluxes) > 1:
            raise Exception("'exp' argument needs to be one of", self.supported_fluxes)

        if len(self.supported_fluxes) == 1:
            return self.__getattribute__(self.supported_fluxes[0])
        else:
            return self.__getattribute__(exp)

    @property
    def quantities(self, exp=""):
        """
        Retrieve the quantities supported by the spline file.

        The default quantities are 'muflux', 'muratio', 'numuflux', 'numuratio',
        'nueflux', 'nueratio', 'flavorratio', 'mu+', 'mu-', 'numu', 'antinumu',
        'nue', 'antinue'. The quantities are the same for all locations. Those with
        'flux' in the names sums over conventional 'mu+' and 'mu-', and neutrino and
        antineutrino, respectively. Those with 'ratio' in the names are the ratios of
        the fluxes. A second set of quantities is available with the 'total_' prefix,
        which includes is a sum of the conventional and prompt fluxes. The latter are
        calculated with the SIBYLL2.3d hadronic interaction model.

        Parameters
        ----------
        exp : str, optional
            Experiment name, default is an empty string.
        """
        return self._get_flux_instance(exp)._quantities

    @property
    def params(self, exp=""):
        """
        Retrieve a list of the daemonflux parameters.

        Parameters
        ----------
        exp : str, optional
            Experiment name, default is an empty string.
        """
        return self._get_flux_instance(exp)._params

    def flux(self, energy, zenith_deg, quantity, params={}, exp=""):
        """
        The flux of a given quantity for the specified energy energy and zenith angles.

        The flux is multiplied by E^3 in units of GeV^2/(cm^2 s sr).

        Parameters
        ----------
        energy : float or np.ndarray
            The energy energy in units of GeV.
        zenith_deg : float or str or list of float
            The zenith angle in degrees. If "average", the average flux over all
            zenith angles will be returned.
        quantity : str
            The type of flux to be returned.
        params : Dict[str, float], optional
            A dictionary of parameter values to shift daemonflux off the baseline.

        Returns
        -------
        np.ndarray
            The flux multiplied by E^3 in units of GeV^2/(cm^2 s sr).

        Raises
        ------
        Exception
            If `zenith_deg` is "average" but splines do not contain average flux.
        """
        return self._get_flux_instance(exp).flux(energy, zenith_deg, quantity, params)

    def error(self, energy, zenith_deg, quantity, only_hadronic=False, exp=""):
        """
        The flux of a given quantity for the specified energy energy and zenith angles.

        The error is multiplied by E^3 in units of GeV^2/(cm^2 s sr).

        Parameters
        ----------
        energy : float or np.ndarray
            The energy energy for which to compute the error.
        zenith_deg : float or str or list of float
            The zenith angle in degrees, or a list of zenith angles.
        quantity : str
            The quantity to compute the error for.
        only_hadronic : bool, optional
            Whether to only include the hadronic error, excluding the cosmic ray flux
            error, by default False.

        Returns
        -------
        np.ndarray
            The error of the flux estimation.

        Raises
        ------
        Exception
            If `zenith_deg` is "average" but splines do not contain average flux.
        """
        return self._get_flux_instance(exp).error(
            energy,
            zenith_deg,
            quantity,
            only_hadronic,
        )

    def chi2(self, params={}, exp=""):
        """
        Returns the chi-square value of the parameters.

        Parameters
        ----------
        params: dict
            Dictionary of modified parameters

        Returns
        -------
        float
            Chi-square value associated with params
        """
        return self._get_flux_instance(exp).chi2(params)

    def __getitem__(self, exp_label):
        if exp_label not in self.supported_fluxes:
            raise KeyError("Supported fluxes are", self.supported_fluxes)
        return self.__getattribute__(exp_label)


class _FluxEntry(Flux):
    _quantities: List[str] = []

    def __init__(
        self,
        label: str,
        fl_spl: dict,
        jac_spl: dict,
        params: Parameters,
        debug: int,
    ) -> None:
        """
        Initialize a `_FluxEntry` object.

        Parameters
        ----------
        label : str
            The label of the flux entry.
        fl_spl : dict
            A dictionary of splines of the flux.
        jac_spl : dict
            A dictionary of splines of the jacobian.
        params : `Parameters`
            A `Parameters` object containing the parameters.
        debug : int
            The debug level.

        Raises
        ------
        AssertionError
            If the splines are not initialized.
        """
        self.label = label
        self._fl_spl = fl_spl
        self._jac_spl = jac_spl
        self._params = params
        self._debug = debug
        assert self._fl_spl is not None, "Splines have to be initialized"
        assert self._jac_spl is not None, "Jacobians required for error estimate"
        self._spl_contains_average = "average" in self._fl_spl
        self._zenith_angles = [a for a in self._fl_spl.keys() if a != "average"]
        self._zenith_deg_arr = np.asarray([float(a) for a in self._zenith_angles])
        self._zenith_deg_arr.sort()
        self._zenith_cos_arr = np.cos(np.deg2rad(self._zenith_deg_arr))
        self._quantities = list(self._fl_spl[list(self._fl_spl.keys())[0]].keys())

    @property
    def zenith_angles(self) -> list:
        """
        Get the list of formatted zenith angles.

        Returns
        -------
        list
            The list of formatted zenith angles.
        """
        return [format_angle(a) for a in self._zenith_deg_arr]

    @contextmanager
    def _temporary_parameters(self, modified_params: dict):
        from copy import deepcopy

        if not modified_params:
            yield
        else:
            prev = deepcopy(self._params)
            pars = self._params
            for k in modified_params:
                if k not in pars.known_parameters:
                    raise KeyError(f"Cannot modify {k}, paramter unknown.")
                par_idx = pars.known_parameters.index(k)
                pars.values[par_idx] += pars.errors[par_idx] * modified_params[k]

            yield
            self._params = prev

    @contextmanager
    def _temporary_exclude_parameters(self, exclude_str):
        from copy import deepcopy

        if not exclude_str:
            yield
        else:
            prev = deepcopy(self._params)
            pars = self._params
            new_kp = []
            new_values = []
            keep_cov = []
            for ik, k in enumerate(pars.known_parameters):
                if exclude_str in k:
                    continue
                new_kp.append(k)
                new_values.append(pars.values[ik])
                keep_cov.append(ik)

            pars.cov = np.take(np.take(pars.cov, keep_cov, axis=0), keep_cov, axis=1)
            pars.values = np.asarray(new_values)
            pars.known_parameters = new_kp

            assert pars.cov.shape[0] == len(pars.known_parameters) == len(pars.values)

            yield

            self._params = prev
            assert pars.cov.shape[0] == len(pars.known_parameters) == len(pars.values)

    def _check_input(self, energy: Union[np.ndarray, float], quantity: str) -> None:
        """
        Check the validity of the input energy and quantity.

        Parameters
        ----------
        energy : float or np.ndarray
            The energy energy.
        quantity : str
            The quantity to be calculated.

        Raises
        ------
        AssertionError
            If the energy energy is out of range or if the quantity is unknown.
        """
        assert np.max(energy) <= 1e9 and np.min(energy) >= 5e-2, "Energy out of range"
        assert quantity in self._quantities, "Quantity must be one of {0}.".format(
            ", ".join(self._quantities)
        )

    def _flux_from_spl(
        self,
        energy: Union[np.ndarray, float],
        zenith_deg: str,
        quantity: str,
        params: Dict[str, float],
    ) -> np.ndarray:
        """
        Calculate the flux based on spline representation.

        Parameters
        ----------
        energy : float or numpy.ndarray
            The grid of energies for which the flux is calculated.
        zenith_deg : str
            The zenith angle in degrees for which the flux is calculated.
        quantity : str
            The type of quantity for which the flux is calculated.
        params : Dict[str, float], optional
            The parameters for the flux calculation.
        """
        jac = self._jac_spl[zenith_deg]
        fl = self._fl_spl[zenith_deg]
        with self._temporary_parameters(params):
            corrections = 1 + np.sum(
                [v * jac[dk][quantity](np.log(energy)) for (dk, v) in self._params],
                axis=0,
            )
        return (np.exp(fl[quantity](np.log(energy))) * corrections).squeeze()

    def _flux_from_interp(
        self,
        energy: Union[np.ndarray, float],
        zenith_deg: Union[float, str, np.ndarray],
        quantity: str,
        params: Dict[str, float],
    ) -> np.ndarray:
        from scipy.interpolate import interp1d

        energy = np.atleast_1d(energy).astype("float64")
        zenith_darr = np.atleast_1d(zenith_deg).astype("float64")
        zenith_carr = np.cos(np.deg2rad(zenith_darr))

        idxmin, idxmax = self._interpolation_domain(zenith_darr)
        interp_array = np.zeros((idxmax - idxmin, len(energy)))
        for i, idx in enumerate(range(idxmin, idxmax)):
            interp_array[i, :] = self._flux_from_spl(
                energy, self.zenith_angles[idx], quantity, params
            )
        return interp1d(self._zenith_cos_arr[idxmin:idxmax], interp_array, axis=0)(
            zenith_carr
        ).T.squeeze()

    def _error_from_spl(
        self,
        energy: Union[np.ndarray, float],
        zenith_deg: Union[float, str, np.ndarray],
        quantity: str,
        only_hadronic: bool,
    ) -> np.ndarray:
        if self._debug > 2:
            print(
                f"Return {quantity} flux for {zenith_deg}, only_hadronic=",
                only_hadronic,
            )

        with self._temporary_exclude_parameters("GSF" if only_hadronic else None):
            jac = self._jac_spl[zenith_deg]

            jacfl = np.vstack(
                [
                    jac[par][quantity](np.log(energy))
                    for par in self._params.known_parameters
                ]
            ).T
            error = np.sqrt(np.diag(grid_cov(jacfl, self._params.cov)))
            return (
                np.exp(self._fl_spl[zenith_deg][quantity](np.log(energy))) * error
            ).squeeze()

    def _error_from_interp(
        self,
        energy: Union[np.ndarray, float],
        zenith_deg: Union[float, str, np.ndarray],
        quantity: str,
        only_hadronic: bool,
    ) -> np.ndarray:
        from scipy.interpolate import interp1d

        energy = np.atleast_1d(energy).astype("float64")
        zenith_darr = np.atleast_1d(zenith_deg).astype("float64")
        zenith_carr = np.cos(np.deg2rad(zenith_darr))

        idxmin, idxmax = self._interpolation_domain(zenith_darr)
        interp_array = np.zeros((idxmax - idxmin, len(energy)))
        for i, idx in enumerate(range(idxmin, idxmax)):
            interp_array[i, :] = self._error_from_spl(
                energy, self.zenith_angles[idx], quantity, only_hadronic
            )
        return interp1d(self._zenith_cos_arr[idxmin:idxmax], interp_array, axis=0)(
            zenith_carr
        ).T.squeeze()

    def _interpolation_domain(
        self, zenith_angles_deg: Union[float, np.ndarray]
    ) -> Tuple[int, int]:
        """Return the indices of `_zenith_deg_arr` that correspond to
        the requested `zenith_angles_deg`

        Args:
            zenith_angles_deg (float or np.ndarray): Zenith angles in degrees

        Returns:
            Tuple[int, int]: The indices of `_zenith_deg_arr` that correspond
            to the requested `zenith_angles_deg`
        """
        zenith_angles_deg = np.atleast_1d(zenith_angles_deg).astype(float)
        if not np.all(np.diff(zenith_angles_deg) >= 0):
            raise ValueError("Requested angles must be sorted in ascending order.")
        if np.min(zenith_angles_deg) < np.min(self._zenith_deg_arr) or np.max(
            zenith_angles_deg
        ) > np.max(self._zenith_deg_arr):
            raise ValueError(
                "Requested zenith angles must be within the range {0} - {1}".format(
                    format_angle(self._zenith_deg_arr[0]),
                    format_angle(self._zenith_deg_arr[-1]),
                )
            )
        if len(self._zenith_angles) < 2:
            raise ValueError("Zenith angle array must have at least two elements.")

        idxmin = np.digitize(np.min(zenith_angles_deg), self._zenith_deg_arr) - 1
        idxmax = min(
            len(self._zenith_deg_arr),
            np.digitize(np.max(zenith_angles_deg), self._zenith_deg_arr) + 1,
        )
        assert self._zenith_deg_arr[idxmax - 1] >= np.max(zenith_angles_deg)
        if idxmax - idxmin < 2:
            idxmax += 1
            if idxmax > len(self._zenith_deg_arr):
                idxmin -= 1
                idxmax -= 1
        return idxmin, idxmax

    def flux(
        self,
        energy: Union[float, np.ndarray],
        zenith_deg: Union[float, str, np.ndarray],
        quantity: str,
        params: Dict[str, float] = {},
    ) -> Union[float, np.ndarray]:
        """
        Compute the flux at the given energy energy, zenith angle, and quantity.

        Parameters
        ----------
        energy : float or np.ndarray
            The energy energy in units of GeV.
        zenith_deg : float or str or list of float
            The zenith angle in degrees. If "average", the average flux over all
            zenith angles will be returned.
        quantity : str
            The type of flux to be returned.
        params : Dict[str, float], optional
            A dictionary of parameter values to shift daemonflux off the baseline.

        Returns
        -------
        float or np.ndarray
            The flux multiplied by E^3 in units of GeV^2/(cm^2 s sr).

        Raises
        ------
        Exception
            If `zenith_deg` is "average" but splines do not contain average flux.
        """
        self._check_input(energy, quantity)
        # handle the case where the zenith angle is "average"
        if isinstance(zenith_deg, str) and zenith_deg == "average":
            if not self._spl_contains_average:
                raise Exception("Splines do not contain average flux")
            return self._flux_from_spl(energy, zenith_deg, quantity, params)

        # handle the case where the zenith angle is a single value or an array
        if not is_iterable(zenith_deg) and float(zenith_deg) in self._zenith_deg_arr:
            return self._flux_from_spl(
                energy, format_angle(float(zenith_deg)), quantity, params
            )
        else:
            return self._flux_from_interp(energy, zenith_deg, quantity, params)

    def error(
        self,
        energy: Union[float, np.ndarray],
        zenith_deg: Union[float, str],
        quantity: str,
        only_hadronic: bool = False,
    ) -> Union[float, np.ndarray]:
        """
        Return the error of the flux estimation for the given parameters.

        The error is multiplied by E^3 in units of GeV^2/(cm^2 s sr).

        Parameters
        ----------
        energy : float or np.ndarray
            The energy energy for which to compute the error.
        zenith_deg : float or str
            The zenith angle in degrees. If a string, it must be "average".
        quantity : str
            The quantity to compute the error for.
        only_hadronic : bool, optional
            Whether to only include the hadronic error, excluding the cosmic ray flux
            error, by default False.

        Returns
        -------
        float or np.ndarray
            The error of the flux estimation.

        Raises
        ------
        Exception
            If `zenith_deg` is "average" but splines do not contain average flux.
        """
        self._check_input(energy, quantity)

        # handle the case where the zenith angle is "average"
        if isinstance(zenith_deg, str) and zenith_deg == "average":
            if not self._spl_contains_average:
                raise Exception("Splines do not contain average flux")
            return self._error_from_spl(energy, zenith_deg, quantity, only_hadronic)

        # handle the case where the zenith angle is a single value or an array
        if not is_iterable(zenith_deg) and float(zenith_deg) in self._zenith_deg_arr:
            return self._error_from_spl(
                energy, format_angle(zenith_deg), quantity, only_hadronic
            )
        else:
            return self._error_from_interp(energy, zenith_deg, quantity, only_hadronic)

    def chi2(self, params={}):
        """
        Returns the chi-square value of the parameters.

        Parameters
        ----------
        params: dict
            Dictionary of modified parameters

        Returns
        -------
        numpy.ndarray
            Chi-square value associated with params
        """
        with self._temporary_parameters(params):
            return self._params.chi2
