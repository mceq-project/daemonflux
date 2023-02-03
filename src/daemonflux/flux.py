import numpy as np
import pickle
import pathlib

# Anatoli's installation requires me to add this
import sys
from scipy.interpolate import fitpack2

sys.modules["scipy.interpolate._fitpack2"] = sys.modules["scipy.interpolate.fitpack2"]

base_path = pathlib.Path(__file__).parent.absolute()


def format_angle(ang):
    return "{:4.4f}".format(float(ang))


class Flux:
    _quantities = [
        "muflux",
        "muratio",
        "numuflux",
        "numuratio",
        "nueflux",
        "nueratio",
        "flavorratio",
        "mu+",
        "mu-",
        "numu",
        "antinumu",
        "nue",
        "antinue",
    ]
    _data_dir = pathlib.Path(base_path, "data")
    _default_spl_version = "20221115"
    _default_cal_version = "20221115_0"

    def __init__(
        self,
        splversion=None,
        calversion=None,
        exclude=[],
    ):  # Python3 stuff -> None:
        # Flux tables naming (JP)
        self._mufit_spl_file = None
        self._calibration_file = None
        self.exclude = exclude

        self.splver = splversion if splversion else self._default_spl_version
        self.calver = calversion if calversion else self._default_cal_version

        self._load_splines()
        self._load_calibration()

        self._apply_little_hacks()

    def _get_grid_cov(self, jac, cov):
        return np.dot(jac, np.dot(cov, jac.T))

    def _apply_little_hacks(self):
        self.GSF19_covinv = np.linalg.inv(self.GSF19_cov)
        self.GSF19_err = np.sqrt(np.diag(self.GSF19_cov))

        for ip, param in enumerate(self.known_params):
            if "uhe_" in param:
                newp = "v" + param[1:]
                print("Renaming param" + param + " to " + newp)
                for location in self.jac_spl:
                    for angle in self.jac_spl[location]:
                        self.jac_spl[location][angle][newp] = self.jac_spl[location][
                            angle
                        ].pop(param)
                self.known_params[ip] = newp

        try:
            self.fl_spl["average"]["0.0000"] = self.fl_spl["average"]["average"]
            self.jac_spl["average"]["0.0000"] = self.jac_spl["average"]["average"]
        except Exception as e:
            print("Did not find an average flux", e)

        self.fl_spl["opera"] = self.fl_spl["ams"]
        self.jac_spl["opera"] = self.jac_spl["ams"]

        if "DEIS" in self.fl_spl:
            print("Adjusting DEIS name")
            self.fl_spl["deis2"] = self.fl_spl["DEIS"]
            self.jac_spl["deis2"] = self.jac_spl["DEIS"]

    def _load_splines(self):
        spl_fname = (
            self._data_dir
            / f"daemon_files_{self.splver}"
            / f"splines_{self.splver}.pkl"
        )
        assert spl_fname.is_file(), f"Calibration file {spl_fname} not found."
        (
            self.known_params,
            self.fl_spl,
            self.jac_spl,
            self.GSF19_cov,
        ) = pickle.load(open(spl_fname, "rb"))

        # Remove combined parameters
        exclude_list = ["pbar", "pi+", "pi-", "K+", "K-", "N"] + self.exclude
        if "he_p" in self.known_params:
            exclude_list += ["p", "n"]
        for p in self.known_params:
            if "pbar" in p:
                exclude_list += [p]

        self.known_params = [k for k in self.known_params if k not in exclude_list]
        print("Excluded params removed from known_params.")

    def _load_calibration(self):
        """Sorts the calibration file contents according to
        the order in known_params."""
        cal_fname = (
            self._data_dir
            / f"daemon_files_{self.splver}"
            / f"calibration_{self.calver}.pkl"
        )
        assert cal_fname.is_file(), f"Calibration file {cal_fname} not found."

        # TODO Remove python2 compatibility
        try:
            calibration_d = pickle.load(
                open(str(cal_fname), "rb")
            )  # , encoding="latin1")
        except:
            calibration_d = pickle.load(open(str(cal_fname), "rb"), encoding="latin1")

        def _remap_covariance(original_order, new_order, cov):
            cov_new = np.zeros((len(new_order), len(new_order)))
            remap = original_order
            for i in range(cov_new.shape[0]):
                for j in range(cov_new.shape[1]):
                    cov_new[i, j] = cov[remap[new_order[i]], remap[new_order[j]]]
            return cov_new

        self.cal_values = {}
        for ip, n in enumerate(self.known_params):
            try:
                self.cal_values[n] = calibration_d["params"][n]["value"]
            except KeyError:
                print("No calibration for", n)
                raise KeyError(n)

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
        self.cal_cov = _remap_covariance(
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
                    == self.cal_cov[ip, jp]
                ), (
                    "Covariance for parameters "
                    + str(pi)
                    + " "
                    + str(pj)
                    + " incorrectly sorted."
                )

        # Get the individual errors from the muon fit
        self.param_errors = np.sqrt(np.diag(self.cal_cov))
        # Invert the covariance matrix
        self.cal_cov_inv = np.linalg.inv(self.cal_cov)

    def _check_input(
        self, grid, exp_tag, angle, params, quantity
    ):  # , use_calibration):
        # assert (
        #    sum((bool(params), use_calibration)) < 2
        # ), "Can not use calibration and tuned params simultaneously."

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
        # use_calibration=False,
        make_checks=True,
    ):
        angle = format_angle(zenith_deg) if zenith_deg != "average" else zenith_deg
        if make_checks:
            self._check_input(
                grid, exp_tag, angle, params, quantity
            )  # , use_calibration)
        # if use_calibration:
        #    params = self.cal_values
        # else:
        # User fits "sigmas" wrt the fit
        # We recast the parameters here
        params = self.recast_params(params)

        # After recasting the parameters, compute their penalty term (chi2)
        # You need to access it with self.chi2_value
        self.get_params_chi2(params)

        jac = self.jac_spl[exp_tag][angle]
        fl = self.fl_spl[exp_tag][angle]

        corrections = 1 + np.sum(
            [
                v * jac[dk][quantity](np.log(grid))
                for (dk, v) in params.items()
                if dk not in self.exclude
            ],
            axis=0,
        )
        return np.exp(fl[quantity](np.log(grid))) * corrections

    def recast_params(self, params):
        new_params = {}
        for ik, onep in enumerate(self.known_params):
            if onep in params:
                new_params[onep] = (
                    self.cal_values[onep] + self.param_errors[ik] * params[onep]
                )
            else:
                new_params[onep] = self.cal_values[onep]

        # print(new_params)

        return new_params

    def get_params_chi2(self, params={}):

        # Placing parameters in a vector with the correct order
        params_array = np.zeros([len(params), 1])
        for ik, onep in enumerate(self.known_params):
            params_array[ik, 0] = params[onep] - self.cal_values[onep]

        params_array = np.matrix(params_array.T)
        self.params_array = params_array

        # Evaluate chi2 using full covariance matrix
        self.chi2_value = (params_array * self.cal_cov_inv * params_array.T)[0, 0]

    def get_error_new(
        self,
        grid,
        exp_tag,
        zenith_deg,
        quantity,
        params={},
        make_checks=True,
        use_calibration=False,
        include_GSF_in_total=True,
    ):

        angle = format_angle(zenith_deg) if zenith_deg != "average" else zenith_deg
        if make_checks:
            self._check_input(grid, exp_tag, angle, params, quantity, use_calibration)

        if len(params) > 0:
            sum_params = params
        elif include_GSF_in_total:
            sum_params = [p for p in self.known_params]
        else:
            sum_params = [p for p in self.known_params if "GSF" not in p]

        jac = self.jac_spl[exp_tag][angle]

        if use_calibration:
            jacfl = np.vstack(
                [jac[par][quantity](np.log(grid)) for par in sum_params]
            ).T
            error = np.sqrt(np.diag(self._get_grid_cov(jacfl, self.cal_cov)))
        else:
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
