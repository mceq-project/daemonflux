import numpy as np
import pickle
import pathlib
from dataclasses import dataclass
from utils import grid_cov, quantities

# # Anatoli's installation requires me to add this
# import sys
# from scipy.interpolate import fitpack2
# sys.modules["scipy.interpolate._fitpack2"] = sys.modules["scipy.interpolate.fitpack2"]


base_path = pathlib.Path(__file__).parent.absolute()


def format_angle(ang):
    return "{:4.4f}".format(float(ang))


@dataclass
class FluxParameters:
    known_parameters: list
    parameter_values: np.ndarray
    cov: np.ndarray

    @property
    def cov_inv(self):
        return np.linalg.inv(self.cov)

    @property
    def param_errors(self):
        return np.sqrt(np.diag(self.cov))

    @property
    def chi2(self, params={}):
        return grid_cov(self.parameter_values, self.cov_inv)
    
    def __iter__(self):
        for p, pv in zip(self.known_parameters, self.parameter_values):
            print("Hello",p, pv)
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

        self.param_defaults = {}
        self.cov = None

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
        
        assert pathlib.Path(spl_file).is_file(), f"Spline file {spl_file} not found."
        (
            self.known_params,
            self.fl_spl,
            self.jac_spl,
            self.GSF19_cov,
        ) = pickle.load(open(spl_file, "rb"))

        self.known_params = [
            k
            for k in self.known_params
            if k not in self.exclude or k.startswith("total_")
        ]

        if cal_file is None:
            # print("daemonflux calibration not used.")
            self.params = FluxParameters(
                self.known_params,
                np.zeros(len(self.known_params)),
                np.diag(np.ones_like(self.known_params)),
            )
            return

        assert isinstance(cal_file, pathlib.Path)
        assert cal_file.is_file(), f"Calibration file {cal_file} not found."

        # TODO Remove python2 compatibility
        # try:
        calibration_d = pickle.load(open(str(cal_file), "rb"))  # , encoding="latin1")
        # except:
        #     calibration_d = pickle.load(open(str(cal_file), "rb"), encoding="latin1")

        def _remap_covariance(original_order, new_order, cov):
            cov_new = np.zeros((len(new_order), len(new_order)))
            remap = original_order
            for i in range(cov_new.shape[0]):
                for j in range(cov_new.shape[1]):
                    cov_new[i, j] = cov[remap[new_order[i]], remap[new_order[j]]]
            return cov_new

        param_values = []
        for ip, n in enumerate(self.known_params):
            try:
                param_values = calibration_d["params"][n]["value"]
            except KeyError:
                raise KeyError("No calibration for", n)

        # Reorder the covariance such that it corresponds to the know_params order
        original_param_order = dict(
            [
                (p, ip)
                for ip, p in enumerate(calibration_d["cov_params"])
                if p in self.known_params
            ]
        )
        n_physics_params = max(original_param_order.values()) + 1

        assert sorted(original_param_order.keys()) == sorted(
            self.known_params
        ), "Parameters inconsistent between spl and calibration file"

        # Create a new covariance with the correct order of parameters
        cov = _remap_covariance(
            original_param_order,
            self.known_params,
            calibration_d["cov_matrix"][:n_physics_params, :n_physics_params],
        )
        for ip, pi in enumerate(self.known_params):
            for jp, pj in enumerate(self.known_params):
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
        self.params = FluxParameters(self.known_params, np.asarray(param_values), cov)

    # def _set_new_parameters(self, params):

    #     assert self.cov is not None, "Covariance needs to be defined."
    #     # Get the individual errors on the parameters from cov
    #     self.param_errors = np.sqrt(np.diag(self.cov))
    #     # Invert the covariance matrix
    #     self.cov_inv = np.linalg.inv(self.cov)

    #     new_defaults = {}
    #     for ip, onep in enumerate(self.known_params):
    #         if onep in params:
    #             new_params[onep] = (
    #                 self.param_defaults[onep] + self.param_errors[ip] * params[onep]
    #             )
    #         else:
    #             new_params[onep] = self.param_defaults[onep]

    #     self.calculate_chi2(params)

    #     return new_params

    def _check_input(self, grid, exp_tag, angle, params, quantity):

        assert self.fl_spl is not None, "Splines have to be initialized"
        assert self.jac_spl is not None, "Jacobians required for error estimate"
        assert np.max(grid) < 1e9 and np.min(grid) >= 5e-2, "Energy out of range"
        assert exp_tag in self.fl_spl, "Available experiments: {0}".format(
            ", ".join(self.fl_spl.keys())
        )
        assert angle in self.fl_spl[exp_tag], "Available angles: {0}".format(
            ", ".join(self.fl_spl[exp_tag].keys())
        )
        for k in params:
            assert k in self.known_params, "Correction parameter {0} not in {1}".format(
                k, ", ".join(self.known_params)
            )
        assert quantity in self._quantities, "Quantity must be one of {0}.".format(
            ", ".join(self._quantities)
        )
        assert (
            sum(["GSF_" in p for p in self.exclude]) == 0
        ), "Individual GSF parameters can't be excluded, use 'GSF' globally."

    def get_flux_new(
        self,
        grid,
        exp_tag,
        zenith_deg,
        quantity,
        params={},
        make_checks=True,
    ):
        angle = format_angle(zenith_deg)  # if zenith_deg != "average" else zenith_deg
        if make_checks:
            self._check_input(grid, exp_tag, angle, params, quantity)

        # if params:
        #     self._set_new_parameters(params)
        # else:
        #     self._reset_params()

        # params = self.recast_params(params)

        jac = self.jac_spl[exp_tag][angle]
        fl = self.fl_spl[exp_tag][angle]

        corrections = 1 + np.sum(
            [
                v * jac[dk][quantity](np.log(grid))
                for (dk, v) in self.params
                if dk not in self.exclude
            ],
            axis=0,
        )
        return np.exp(fl[quantity](np.log(grid))) * corrections

    def get_error_new(
        self,
        grid,
        exp_tag,
        zenith_deg,
        quantity,
        params={},
        make_checks=True,
        include_GSF_in_total=True,
    ):

        angle = format_angle(zenith_deg) if zenith_deg != "average" else zenith_deg
        if make_checks:
            self._check_input(grid, exp_tag, angle, params, quantity)

        if len(params) > 0:
            sum_params = params
        elif include_GSF_in_total:
            sum_params = [p for p in self.known_params]
        else:
            sum_params = [p for p in self.known_params if "GSF" not in p]

        jac = self.jac_spl[exp_tag][angle]

        jacfl = np.vstack(
            [jac[par][quantity](np.log(grid)) for par in sum_params]
        ).T
        error = np.sqrt(np.diag(self._get_grid_cov(jacfl, self.cov)))
        # if use_calibration:
        # else:
        error = np.sqrt(
            np.sum(
                [jac[dk][quantity](np.log(grid)) ** 2 for dk in sum_params],
                axis=0,
            )
        )

        if "GSF" not in self.exclude:
            jacfl = np.vstack(
                # [jac[f"GSF_{i}"][quantity](np.log(grid)) for i in range(1, 7)]
                # ).T
                [
                    jac["GSF_" + "%i" % i][quantity](np.log(grid))
                    for i in range(1, 7)
                ]
            ).T
            gsf_error = np.sqrt(np.diag(self._get_grid_cov(jacfl, self.GSF19_cov)))
            error = np.sqrt(error**2 + gsf_error**2)

        return np.exp(self.fl_spl[exp_tag][angle][quantity](np.log(grid))) * error

    def print_experiments(self):
        # if self.fl_spl is None and self.jac_spl is None:
        if self.fl_spl is None:
            raise RuntimeError("Can not print experiments if flux spline unassigned.")
        for exp in self.fl_spl:
            print("{0}: [{1}]".format(exp, ", ".join(self.fl_spl[exp].keys())))
