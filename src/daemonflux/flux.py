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
        self.values = values
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
        return np.sum(grid_cov(self.values, self.invcov))

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
    _default_url = (
        "https://github.com/mceq-project/daemonflux/releases/download/v0.4.1/"
    )
    _default_spl_file = "daemonsplines_{0}_202303_0.pkl"
    _default_cal_file = "daemonsplines_calibration_202303_0.pkl"

    def __init__(
        self,
        location="generic",
        spl_file=None,
        cal_file=None,
        use_calibration=True,
        exclude=[],
        debug=1,
    ) -> None:
        self.exclude = exclude
        self._debug = debug
        spl_file = (
            spl_file
            if spl_file
            else _cached_data_dir(
                self._default_url + self._default_spl_file.format(location)
            )
        )
        cal_file = (
            cal_file
            if cal_file
            else _cached_data_dir(self._default_url + self._default_cal_file)
        )

        if not use_calibration:
            cal_file = None

        self._load_splines(spl_file, cal_file)

    def _load_splines(self, spl_file, cal_file):
        from .utils import rearrange_covariance
        from copy import deepcopy

        assert pathlib.Path(spl_file).is_file(), f"Spline file {spl_file} not found."
        (
            known_pars,
            self._fl_spl,
            self._jac_spl,
            cov,
        ) = pickle.load(open(spl_file, "rb"))

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

            calibration_d = pickle.load(open(str(cal_file), "rb"), encoding="latin1")

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

    def print_experiments(self):
        for exp in self._fl_spl:
            print("{0}: [{1}]".format(exp, ", ".join(self._fl_spl[exp].keys())))

    @property
    def zenith_angles(self, exp=""):
        if not exp and len(self.supported_fluxes) > 1:
            raise Exception("'exp' argument needs to be one of", self.supported_fluxes)
        if len(self.supported_fluxes) == 1:
            return self.__getattribute__(self.supported_fluxes[0]).zenith_angles
        else:
            return self.__getattribute__(exp).zenith_angles

    def _get_flux_instance(self, exp):
        if not exp and len(self.supported_fluxes) > 1:
            raise Exception("'exp' argument needs to be one of", self.supported_fluxes)

        if len(self.supported_fluxes) == 1:
            return self.__getattribute__(self.supported_fluxes[0])
        else:
            return self.__getattribute__(exp)

    @property
    def quantities(self, exp=""):
        return self._get_flux_instance(exp)._quantities

    @property
    def params(self, exp=""):
        return self._get_flux_instance(exp)._params

    def flux(self, grid, zenith_deg, quantity, params={}, exp=""):
        return self._get_flux_instance(exp).flux(grid, zenith_deg, quantity, params)

    def error(self, grid, zenith_deg, quantity, only_hadronic=False, exp=""):
        return self._get_flux_instance(exp).error(
            grid,
            zenith_deg,
            quantity,
            only_hadronic,
        )

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

    def _check_input(self, grid: np.ndarray, quantity: str) -> None:
        """
        Check the validity of the input grid and quantity.

        Parameters
        ----------
        grid : np.ndarray
            The energy grid.
        quantity : str
            The quantity to be calculated.

        Raises
        ------
        AssertionError
            If the energy grid is out of range or if the quantity is unknown.
        """
        assert np.max(grid) <= 1e9 and np.min(grid) >= 5e-2, "Energy out of range"
        assert quantity in self._quantities, "Quantity must be one of {0}.".format(
            ", ".join(self._quantities)
        )

    def _flux_from_spl(
        self,
        grid: np.ndarray,
        zenith_deg: str,
        quantity: str,
        params: Dict[str, float],
    ) -> np.ndarray:
        """
        Calculate the flux based on spline representation.

        Parameters
        ----------
        grid : numpy.ndarray
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
                [v * jac[dk][quantity](np.log(grid)) for (dk, v) in self._params],
                axis=0,
            )
        return np.exp(fl[quantity](np.log(grid))) * corrections

    def _flux_from_interp(
        self,
        grid: np.ndarray,
        zenith_deg: Union[float, str, np.ndarray],
        quantity: str,
        params: Dict[str, float],
    ) -> np.ndarray:
        from scipy.interpolate import interp1d

        zenith_darr = np.atleast_1d(zenith_deg).astype("float64")
        zenith_carr = np.cos(np.deg2rad(zenith_darr))

        idxmin, idxmax = self._interpolation_domain(zenith_darr)
        interp_array = np.zeros((idxmax - idxmin, len(grid)))
        for i, idx in enumerate(range(idxmin, idxmax)):
            interp_array[i, :] = self._flux_from_spl(
                grid, self.zenith_angles[idx], quantity, params
            )
        return interp1d(self._zenith_cos_arr[idxmin:idxmax], interp_array, axis=0)(
            zenith_carr
        ).T.squeeze()

    def _error_from_spl(
        self,
        grid: np.ndarray,
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
                    jac[par][quantity](np.log(grid))
                    for par in self._params.known_parameters
                ]
            ).T
            error = np.sqrt(np.diag(grid_cov(jacfl, self._params.cov)))
            return np.exp(self._fl_spl[zenith_deg][quantity](np.log(grid))) * error

    def _error_from_interp(
        self,
        grid: np.ndarray,
        zenith_deg: Union[float, str, np.ndarray],
        quantity: str,
        only_hadronic: bool,
    ) -> np.ndarray:
        from scipy.interpolate import interp1d

        zenith_darr = np.atleast_1d(zenith_deg).astype("float64")
        zenith_carr = np.cos(np.deg2rad(zenith_darr))

        idxmin, idxmax = self._interpolation_domain(zenith_darr)
        interp_array = np.zeros((idxmax - idxmin, len(grid)))
        for i, idx in enumerate(range(idxmin, idxmax)):
            interp_array[i, :] = self._error_from_spl(
                grid, self.zenith_angles[idx], quantity, only_hadronic
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
        idxmax = np.digitize(np.max(zenith_angles_deg), self._zenith_deg_arr)
        if idxmax - idxmin < 2:
            idxmax += 1
            if idxmax > len(self._zenith_deg_arr):
                idxmin -= 1
                idxmax -= 1
        return idxmin, idxmax

    def flux(
        self,
        grid: np.ndarray,
        zenith_deg: Union[float, str, np.ndarray],
        quantity: str,
        params: Dict[str, float] = {},
    ) -> np.ndarray:
        """
        Compute the flux at the given energy grid, zenith angle, and quantity.

        Parameters
        ----------
        grid : np.ndarray
            The energy grid in units of GeV.
        zenith_deg : float or str
            The zenith angle in degrees. If "average", the average flux over all
            zenith angles will be returned.
        quantity : str
            The type of flux to be returned.
        params : Dict[str, float], optional
            A dictionary of parameter values to use for the calculation.

        Returns
        -------
        np.ndarray
            The flux multiplied by E^3 in units of GeV^2/(cm^2 s sr).

        Raises
        ------
        Exception
            If `zenith_deg` is "average" but splines do not contain average flux.
        """
        self._check_input(grid, quantity)
        # handle the case where the zenith angle is "average"
        if isinstance(zenith_deg, str) and zenith_deg == "average":
            if not self._spl_contains_average:
                raise Exception("Splines do not contain average flux")
            return self._flux_from_spl(grid, zenith_deg, quantity, params)

        # handle the case where the zenith angle is a single value or an array
        if not is_iterable(zenith_deg) and float(zenith_deg) in self._zenith_deg_arr:
            return self._flux_from_spl(
                grid, format_angle(float(zenith_deg)), quantity, params
            )
        else:
            return self._flux_from_interp(grid, zenith_deg, quantity, params)

    def error(
        self,
        grid: np.ndarray,
        zenith_deg: Union[float, str],
        quantity: str,
        only_hadronic: bool = False,
    ) -> np.ndarray:
        """
        Return the error of the flux estimation for the given parameters.

        The error is multiplied by E^3 in units of GeV^2/(cm^2 s sr).

        Parameters
        ----------
        grid : np.ndarray
            The energy grid for which to compute the error.
        zenith_deg : float or str
            The zenith angle in degrees. If a string, it must be "average".
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
        self._check_input(grid, quantity)

        # handle the case where the zenith angle is "average"
        if isinstance(zenith_deg, str) and zenith_deg == "average":
            if not self._spl_contains_average:
                raise Exception("Splines do not contain average flux")
            return self._error_from_spl(grid, zenith_deg, quantity, only_hadronic)

        # handle the case where the zenith angle is a single value or an array
        if not is_iterable(zenith_deg) and float(zenith_deg) in self._zenith_deg_arr:
            return self._error_from_spl(
                grid, format_angle(zenith_deg), quantity, only_hadronic
            )
        else:
            return self._error_from_interp(grid, zenith_deg, quantity, only_hadronic)
