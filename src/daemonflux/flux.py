import numpy as np
import pickle
import pathlib
from .utils import grid_cov, quantities
from contextlib import contextmanager

# # Anatoli's installation requires me to add this
# import sys
# from scipy.interpolate import fitpack2
# sys.modules["scipy.interpolate._fitpack2"] = sys.modules["scipy.interpolate.fitpack2"]


base_path = pathlib.Path(__file__).parent.absolute()


def format_angle(ang):
    return "{:4.4f}".format(float(ang))


class FluxParameters:
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
    ):  # Python3 stuff -> None:
        # Flux tables naming (JP)
        self._mufit_spl_file = None
        self._calibration_file = None
        self.exclude = exclude

        spl_file = spl_file if spl_file else self._default_spl_file
        cal_file = cal_file if cal_file else self._default_cal_file

        if not use_calibration:
            cal_file = None

        self._load_splines(spl_file, cal_file)

        self._apply_little_hacks()

    def _get_grid_cov(self, jac, cov):
        return np.dot(jac, np.dot(cov, jac.T))

    def _apply_little_hacks(self):
        # self.GSF19_covinv = np.linalg.inv(self.GSF19_cov)
        # self.GSF19_err = np.sqrt(np.diag(self.GSF19_cov))

        self.fl_spl["opera"] = self.fl_spl["ams"]
        self.jac_spl["opera"] = self.jac_spl["ams"]

        # if "DEIS" in self.fl_spl:
        #     print("Adjusting DEIS name")
        #     self.fl_spl["deis2"] = self.fl_spl["DEIS"]
        #     self.jac_spl["deis2"] = self.jac_spl["DEIS"]

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
            # print("daemonflux calibration not used.")
            self.params = FluxParameters(
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

        self.params = FluxParameters(known_parameters, np.asarray(param_values), cov)

    def _check_input(self, grid, exp_tag, angle, quantity):

        assert self.fl_spl is not None, "Splines have to be initialized"
        assert self.jac_spl is not None, "Jacobians required for error estimate"
        assert np.max(grid) <= 1e9 and np.min(grid) >= 5e-2, "Energy out of range"
        assert exp_tag in self.fl_spl, "Available experiments: {0}".format(
            ", ".join(self.fl_spl.keys())
        )
        assert angle in self.fl_spl[exp_tag], "Available angles: {0}".format(
            ", ".join(self.fl_spl[exp_tag].keys())
        )
        assert quantity in self._quantities, "Quantity must be one of {0}.".format(
            ", ".join(self._quantities)
        )
        assert (
            sum(["GSF_" in p for p in self.exclude]) == 0
        ), "Individual GSF parameters can't be excluded, use 'GSF' globally."

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

    def flux(
        self,
        grid,
        exp_tag,
        zenith_deg,
        quantity,
        params={},
        make_checks=True,
    ):
        angle = "average" if zenith_deg == "average" else format_angle(zenith_deg)

        if make_checks:
            self._check_input(grid, exp_tag, angle, quantity)

        jac = self.jac_spl[exp_tag][angle]
        fl = self.fl_spl[exp_tag][angle]
        with self._temporary_parameters(params):
            corrections = 1 + np.sum(
                [v * jac[dk][quantity](np.log(grid)) for (dk, v) in self.params],
                axis=0,
            )
        return np.exp(fl[quantity](np.log(grid))) * corrections

    def error(
        self,
        grid,
        exp_tag,
        zenith_deg,
        quantity,
        make_checks=True,
        only_hadronic=False,
    ):

        angle = "average" if zenith_deg == "average" else format_angle(zenith_deg)

        if make_checks:
            self._check_input(grid, exp_tag, angle, quantity)

        with self._temporary_exclude_parameters("GSF" if only_hadronic else None):
            print(self.params.known_parameters, self.params.values, self.params.errors)

            jac = self.jac_spl[exp_tag][angle]

            jacfl = np.vstack(
                [
                    jac[par][quantity](np.log(grid))
                    for par in self.params.known_parameters
                ]
            ).T
            error = np.sqrt(np.diag(self._get_grid_cov(jacfl, self.params.cov)))
            return np.exp(self.fl_spl[exp_tag][angle][quantity](np.log(grid))) * error

    def print_experiments(self):
        for exp in self.fl_spl:
            print("{0}: [{1}]".format(exp, ", ".join(self.fl_spl[exp].keys())))
