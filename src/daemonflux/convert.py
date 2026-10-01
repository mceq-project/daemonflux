"""Convert legacy pickled spline libraries to version-2 response files.

Species (mu+, mu-, numu, antinumu, nue, antinue and their pr_/total_
variants) are converted, and sums and ratios that the evaluator can form from
them are dropped. A quantity without its species (older experiment profiles
store only muflux and muratio) is kept as its own table, with the legacy
linearized response of that quantity. Legacy
splines are linear interpolants (k=1, s=0) in log energy, so their knots and
values are the original tables and the conversion is lossless with
``energy_interpolation="linear"``.

Usage::

    python -m daemonflux.convert LIB.pkl OUTDIR [--group NAME=PROF1,PROF2 ...]
        [--build TAG] [--nucleons additive|factor] [--interpolation linear|cubic]
"""

import argparse
import pathlib
import pickle

import numpy as np

from .flux import _unpack_extra
from .metadata import file_sha256
from .response import PREFIXES, SPECIES, default_derived, write_response_file

SPECIES_NAMES = {p + s for p in PREFIXES for s in SPECIES}
NUCLEONS = (2212, 2112)
# Large provenance blocks that belong to the generator run, not to each file.
_DROPPED_KEYS = ("parameters", "cases", "computed_domains", "campaign_identity", "merge")


def _table(spline):
    k = spline._data[5] if hasattr(spline, "_data") else None
    if k != 1:
        raise ValueError("Only linear (k=1) interpolating splines convert losslessly")
    knots = spline.get_knots()
    return knots, spline(knots)


def legacy_parameters(names, metadata, nucleons="additive"):
    """Version-2 parameter descriptions for a legacy library."""
    if metadata is None:
        parameters = [
            {
                "name": n,
                "group": "primary" if "GSF" in n else "hadronic",
                "units": "sigma",
            }
            for n in names
        ]
    else:
        parameters = [dict(p) for p in metadata["parameters"]]
    for p in parameters:
        nucleon = p.get("secondary") in NUCLEONS and p["group"] == "hadronic"
        p["combination"] = "factor" if (nucleons == "factor" and nucleon) else "additive"
        p["transform"] = "linear"
    return parameters


def convert_legacy(
    library,
    output_dir,
    groups=None,
    build="converted",
    nucleons="additive",
    interpolation="linear",
):
    """Write one response file per group of profiles; return {path: sha256}.

    ``groups`` maps a dataset id to its profiles; by default each profile is its
    own dataset.
    """
    library = pathlib.Path(library)
    with open(library, "rb") as stream:
        payload = pickle.load(stream)
    names, fl_spl, jac_spl, cov = payload[:4]
    primary_cov = payload[4] if len(payload) > 4 else None
    metadata, linear, height = _unpack_extra(payload[5] if len(payload) > 5 else None)
    if height is not None:
        raise ValueError("Height-grid libraries are not converted")
    parameters = legacy_parameters(names, metadata, nucleons)
    source = {"file": library.name, "sha256": file_sha256(library)}
    shared = {k: v for k, v in (metadata or {}).items() if k not in _DROPPED_KEYS}
    groups = groups or {p: [p] for p in fl_spl}
    written = {}
    for dataset, profiles in groups.items():
        missing = set(profiles) - set(fl_spl)
        if missing:
            raise KeyError(f"Profiles not in {library.name}: {sorted(missing)}")
        tables, profile_meta = {}, {}
        for profile in profiles:
            tables[profile] = {}
            for angle, quantities in fl_spl[profile].items():
                species = {}
                present = set(quantities) & SPECIES_NAMES
                derivable = set(default_derived(present))
                for q, spline in quantities.items():
                    if (linear and q in linear) or q.endswith("_pol"):
                        continue
                    if q not in SPECIES_NAMES and q in derivable:
                        continue
                    x, y = _table(spline)
                    jac = np.vstack(
                        [_table(jac_spl[profile][angle][n][q])[1] for n in names]
                    )
                    species[q] = {"x": np.exp(x), "value": np.exp(y), "jacobian": jac}
                if not species:
                    raise ValueError(f"{profile}/{angle} has no quantities to convert")
                tables[profile][angle] = species
            profile_meta[profile] = {
                "cases": [
                    c
                    for c in (metadata or {}).get("cases", [])
                    if c.get("profile") == profile
                ],
                "computed_domains": (metadata or {})
                .get("computed_domains", {})
                .get(profile),
            }
        meta = dict(shared)
        meta.update(
            parameters=parameters,
            dataset_id=dataset,
            build_id=f"{dataset}_{build}",
            energy_interpolation=interpolation,
            converted_from=source,
        )
        if metadata is None:
            meta["calibration_binding"] = "name"
        path = pathlib.Path(output_dir) / f"daemonflux_{dataset}_{build}.h5"
        written[str(path)] = write_response_file(
            path,
            meta,
            tables,
            covariance=cov,
            primary_covariance=primary_cov,
            profile_metadata=profile_meta,
        )
    return written


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("library")
    parser.add_argument("output_dir")
    parser.add_argument("--group", action="append", default=[], metavar="NAME=P1,P2")
    parser.add_argument("--build", default="converted")
    parser.add_argument("--nucleons", choices=("additive", "factor"), default="additive")
    parser.add_argument("--interpolation", choices=("linear", "cubic"), default="linear")
    args = parser.parse_args(argv)
    groups = None
    if args.group:
        groups = {g.split("=")[0]: g.split("=")[1].split(",") for g in args.group}
    for path, sha in convert_legacy(
        args.library,
        args.output_dir,
        groups,
        args.build,
        args.nucleons,
        args.interpolation,
    ).items():
        print(sha, path)


if __name__ == "__main__":
    main()
