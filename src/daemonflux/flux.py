import numpy as np
import pickle
import pathlib
from .utils import grid_cov, quantities
from contextlib import contextmanager
from abc import ABCMeta

# # Anatoli's installation requires me to add this
# import sys
# from scipy.interpolate import fitpack2
# sys.modules["scipy.interpolate._fitpack2"] = sys.modules["scipy.interpolate.fitpack2"]


base_path = pathlib.Path(__file__).parent.absolute()


def format_angle(ang) -> str:
    return "{:4.4f}".format(float(ang))


class Parameters:
    def __init__(
        self, known_parameters: list, values: np.ndarray, cov: np.ndarray
    ) -> None:
        self.known_parameters = known_parameters
        self.values = values
        self.cov = cov

    @property
    def invcov(self) -> np.ndarray:
        return np.linalg.inv(self.cov)

    @property
    def errors(self) -> np.ndarray:
        return np.sqrt(np.diag(self.cov))

    @property
    def chi2(self, params={}) -> np.ndarray:
        return grid_cov(self.values, self.invcov)

    def __iter__(self):
        for p, pv in zip(self.known_parameters, self.values):
            yield p, pv


class Flux:
    _quantities = quantities
    _data_dir = pathlib.Path(base_path, "data")
    _default_spl_file = _data_dir / "daemonspl_20221115.pkl"
    _default_cal_file = _data_dir / "daemoncal_20221115_0.pkl"

    def __init__(
        self,
        spl_file=None,
        cal_file=None,
        use_calibration=True,
        exclude=[],
    ) -> None:
        self.exclude = exclude

        spl_file = spl_file if spl_file else self._default_spl_file
        cal_file = cal_file if cal_file else self._default_cal_file

        if not use_calibration:
            cal_file = None

        self._load_splines(spl_file, cal_file)

    def _get_grid_cov(self, jac, cov):
        return np.dot(jac, np.dot(cov, jac.T))

    def _load_splines(self, spl_file, cal_file):
        from .utils import rearrange_covariance

        assert pathlib.Path(spl_file).is_file(), f"Spline file {spl_file} not found."
        (
            known_pars,
            self.fl_spl,
            self.jac_spl,
            self.GSF19_cov,
        ) = pickle.load(open(spl_file, "rb"))

        known_parameters = []
        for k in known_pars:
            if k in self.exclude:
                continue
            if k.startswith("total_"):
                continue
            known_parameters.append(k)

        if cal_file is None:
            print("daemonflux calibration not used.")
            self.params = Parameters(
                known_parameters,
                np.zeros(len(known_parameters)),
                np.diag(np.ones(len(known_parameters))),
            )
            return

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

        self.params = Parameters(known_parameters, np.asarray(param_values), cov)
        self.supported_fluxes = []
        for exp in self.fl_spl:
            subflux = FluxEntry(
                exp, self.fl_spl[exp], self.jac_spl[exp], self.params, self.exclude
            )
            setattr(self, exp, subflux)
            self.supported_fluxes.append(exp)

    def print_experiments(self):
        for exp in self.fl_spl:
            print("{0}: [{1}]".format(exp, ", ".join(self.fl_spl[exp].keys())))

    @property
    def zenith_angles(self, exp=""):
        if not exp and len(self.supported_fluxes) > 1:
            raise Exception("'exp' argument needs to be one of", self.supported_fluxes)
        if len(self.supported_fluxes) == 1:
            return self.__getattribute__(self.supported_fluxes[0]).zenith_angles
        else:
            return self.__getattribute__(exp).zenith_angles

    def flux(self, grid, zenith_deg, quantity, params={}, exp=""):
        if not exp and len(self.supported_fluxes) > 1:
            raise Exception("'exp' argument needs to be one of", self.supported_fluxes)
        if len(self.supported_fluxes) == 1:
            return self.__getattribute__(self.supported_fluxes[0]).flux(
                grid, zenith_deg, quantity, params
            )
        else:
            return self.__getattribute__(exp).flux(grid, zenith_deg, quantity, params)

    def error(self, grid, zenith_deg, quantity, only_hadronic=False, exp=""):
        if not exp and len(self.supported_fluxes) > 1:
            raise Exception("'exp' argument needs to be one of", self.supported_fluxes)

        if len(self.supported_fluxes) == 1:
            return self.__getattribute__(self.supported_fluxes[0]).error(
                grid,
                zenith_deg,
                quantity,
                only_hadronic,
            )
        else:
            return self.__getattribute__(exp).error(
                grid,
                zenith_deg,
                quantity,
                only_hadronic,
            )

    def __getitem__(self, exp_label):
        if exp_label not in self.supported_fluxes:
            raise KeyError("Supported fluxes are", self.supported_fluxes)
        return self.__getattribute__(exp_label)

    @contextmanager
    def _temporary_parameters(self, modified_params: dict):
        from copy import copy

        if not modified_params:
            yield
        else:
            prev = copy(self.params.values)
            pars = self.params
            for k in modified_params:
                if k not in pars.known_parameters:
                    raise KeyError(f"Cannot modify {k}, paramter unknown.")
                par_idx = pars.known_parameters.index(k)
                pars.values[par_idx] += pars.errors[par_idx] * modified_params[k]

            yield
            self.params.values = prev

    @contextmanager
    def _temporary_exclude_parameters(self, exclude_str):
        from copy import deepcopy

        if not exclude_str:
            yield
        else:
            prev = deepcopy(self.params)
            pars = self.params
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

            self.params = prev


class FluxEntry(Flux):
    def __init__(self, label, fl_spl, jac_spl, params, exclude) -> None:
        self.label = label
        self.fl_spl = fl_spl
        self.jac_spl = jac_spl
        self.params = params
        self.exclude = exclude
        assert self.fl_spl is not None, "Splines have to be initialized"
        assert self.jac_spl is not None, "Jacobians required for error estimate"

    @property
    def zenith_angles(self):
        return sorted(self.fl_spl.keys())

    def _check_input(self, grid, angle, quantity):

        assert np.max(grid) <= 1e9 and np.min(grid) >= 5e-2, "Energy out of range"
        assert angle in self.zenith_angles, "Available angles: {0}".format(
            ", ".join(self.fl_spl.keys())
        )
        assert quantity in self._quantities, "Quantity must be one of {0}.".format(
            ", ".join(self._quantities)
        )
        assert (
            sum(["GSF_" in p for p in self.exclude]) == 0
        ), "Individual GSF parameters can't be excluded, use 'GSF' globally."

    def flux(
        self,
        grid,
        zenith_deg,
        quantity,
        params={},
    ):
        angle = "average" if zenith_deg == "average" else format_angle(zenith_deg)

        self._check_input(grid, angle, quantity)

        jac = self.jac_spl[angle]
        fl = self.fl_spl[angle]
        with self._temporary_parameters(params):
            corrections = 1 + np.sum(
                [v * jac[dk][quantity](np.log(grid)) for (dk, v) in self.params],
                axis=0,
            )
        return np.exp(fl[quantity](np.log(grid))) * corrections

    def error(
        self,
        grid,
        zenith_deg,
        quantity,
        only_hadronic=False,
    ):
        angle = "average" if zenith_deg == "average" else format_angle(zenith_deg)

        self._check_input(grid, angle, quantity)

        with self._temporary_exclude_parameters("GSF" if only_hadronic else None):

            jac = self.jac_spl[angle]

            jacfl = np.vstack(
                [
                    jac[par][quantity](np.log(grid))
                    for par in self.params.known_parameters
                ]
            ).T
            error = np.sqrt(np.diag(self._get_grid_cov(jacfl, self.params.cov)))
            return np.exp(self.fl_spl[angle][quantity](np.log(grid))) * error
