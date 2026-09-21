"""Optional, versioned metadata for response libraries and their calibrations."""

import hashlib

import numpy as np


def file_sha256(path):
    """Return the identity of a spline artifact without loading it into memory."""
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_metadata(metadata, names):
    """Validate the optional sixth spline payload item; preserve legacy files."""
    if metadata is None:
        return None
    if not isinstance(metadata, dict) or metadata.get("schema_version") != 1:
        raise ValueError("Unsupported spline metadata schema")
    parameters = metadata.get("parameters", [])
    if not all(isinstance(p, dict) for p in parameters):
        raise ValueError("Parameter metadata must contain dictionaries")
    labels = [p.get("name") for p in parameters]
    if labels != list(names) or len(set(labels)) != len(labels):
        raise ValueError("Metadata parameter order must match the spline payload")
    for parameter in parameters:
        if parameter.get("group") not in ("hadronic", "primary"):
            raise ValueError("Each parameter needs a hadronic or primary group")
        if parameter.get("units") not in ("fractional", "sigma"):
            raise ValueError("Each parameter needs fractional or sigma units")
    return metadata


def validate_covariance(covariance, size):
    """Accept an absent prior, or a finite symmetric positive-semidefinite matrix."""
    if covariance is None:
        return None
    cov = np.asarray(covariance, dtype=float)
    if cov.shape != (size, size) or not np.isfinite(cov).all():
        raise ValueError("Parameter covariance has invalid shape or values")
    if not np.allclose(cov, cov.T, rtol=1e-10, atol=1e-14):
        raise ValueError("Parameter covariance must be symmetric")
    scale = max(float(np.max(np.abs(cov))), np.finfo(float).tiny)
    if np.linalg.eigvalsh(cov).min(initial=0) < -1e-10 * scale:
        raise ValueError("Parameter covariance must be positive semidefinite")
    return cov.copy()


def validate_calibration_identity(calibration, spline_path, required=False):
    """Bind a new calibration to one or more exact spline artifact hashes."""
    expected = calibration.get("spline_sha256")
    if expected is None:
        if required:
            raise ValueError("Metadata-bearing splines require calibration spline_sha256")
        return
    accepted = [expected] if isinstance(expected, str) else expected
    if not isinstance(accepted, list) or file_sha256(spline_path) not in accepted:
        raise ValueError("Calibration does not match this spline artifact")
