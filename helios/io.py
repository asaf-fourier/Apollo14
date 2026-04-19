"""Persist a run's inputs to disk for later replay.

A "run" is everything needed to re-execute a trace from scratch: the git
SHA of apollo14, the optical system, the projector, and the scan config.
Assumes the working tree is clean — runs are produced by automated jobs
from committed code.

Serialized as JSON with floats rounded to 6 significant figures for a
compact, human-readable, browser-parseable artifact.
"""

import json
import math
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np

import apollo14
from apollo14.projector import Projector, FovGrid
from apollo14.system import OpticalSystem


FLOAT_SIGFIGS = 6


@dataclass
class ScanConfig:
    base_direction: jnp.ndarray  # (3,)
    x_fov: float                 # radians
    y_fov: float                 # radians
    num_x: int
    num_y: int


def _round_sig(x: float, sigfigs: int = FLOAT_SIGFIGS) -> float:
    if not math.isfinite(x) or x == 0.0:
        return float(x)
    digits = sigfigs - int(math.floor(math.log10(abs(x)))) - 1
    return round(float(x), digits)


def _to_list(a) -> Any:
    arr = np.asarray(a).tolist()

    def walk(v):
        if isinstance(v, list):
            return [walk(x) for x in v]
        if isinstance(v, float):
            return _round_sig(v)
        return v

    return walk(arr)


def _git_sha() -> str:
    repo = Path(apollo14.__file__).resolve().parent.parent
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip()


def _serialize_projector(p: Projector) -> dict:
    out = {
        "position": _to_list(p.position),
        "direction": _to_list(p.direction),
        "beam_width": _round_sig(float(p.beam_width)),
        "beam_height": _round_sig(float(p.beam_height)),
        "nx": int(p.nx),
        "ny": int(p.ny),
        "falloff_x": _round_sig(float(p.falloff_x)),
        "falloff_y": _round_sig(float(p.falloff_y)),
        "spectrum": None,
    }
    if p.spectrum is not None:
        wls, rad = p.spectrum
        out["spectrum"] = {
            "wavelengths": _to_list(wls),
            "radiance": _to_list(rad),
        }
    return out


def _serialize_element(e) -> dict:
    cls = type(e).__name__
    base = {"type": cls, "name": getattr(e, "name", None)}

    if cls in ("PartialMirror", "GaussianMirror"):
        base.update({
            "position": _to_list(e.position),
            "normal": _to_list(e.normal),
            "width": _round_sig(float(e.width)),
            "height": _round_sig(float(e.height)),
            "wavelengths": _to_list(e.wavelengths),
            "reflectance": _to_list(e.reflectance),
        })
        if cls == "GaussianMirror":
            base.update({
                "amplitude": _to_list(e.amplitude),
                "sigma": _to_list(e.sigma),
            })
    elif cls == "RectangularAperture":
        base.update({
            "position": _to_list(e.position),
            "normal": _to_list(e.normal),
            "width": _round_sig(float(e.width)),
            "height": _round_sig(float(e.height)),
            "inner_width": _round_sig(float(e.inner_width)),
            "inner_height": _round_sig(float(e.inner_height)),
        })
    elif cls == "RectangularPupil":
        base.update({
            "position": _to_list(e.position),
            "normal": _to_list(e.normal),
            "width": _round_sig(float(e.width)),
            "height": _round_sig(float(e.height)),
        })
    elif cls == "GlassBlock":
        base.update({
            "position": _to_list(e.position),
            "material": getattr(e.material, "name", None),
            "faces": [
                {
                    "name": f.name,
                    "position": _to_list(f.position),
                    "normal": _to_list(f.normal),
                    "vertices": _to_list(f.vertices),
                }
                for f in e.faces
            ],
        })
    else:
        raise ValueError(f"No serializer for element type {cls!r}")
    return base


def _serialize_system(sys: OpticalSystem) -> dict:
    return {
        "env_material": getattr(sys.env_material, "name", None),
        "elements": [_serialize_element(e) for e in sys.elements],
    }


def _serialize_scan(scan: ScanConfig) -> dict:
    return {
        "base_direction": _to_list(scan.base_direction),
        "x_fov": _round_sig(float(scan.x_fov)),
        "y_fov": _round_sig(float(scan.y_fov)),
        "num_x": int(scan.num_x),
        "num_y": int(scan.num_y),
    }


def save_run(
    run_dir: Path | str,
    system: OpticalSystem,
    projector: Projector,
    scan: ScanConfig,
    *,
    response: jnp.ndarray | None = None,
    pupil_x_mm: jnp.ndarray | None = None,
    pupil_y_mm: jnp.ndarray | None = None,
    scan_angles: jnp.ndarray | None = None,
    wavelengths_nm: jnp.ndarray | None = None,
    extra: dict | None = None,
) -> Path:
    """Write a run's inputs (and optionally results) to ``run_dir``.

    Inputs go to ``manifest.json`` (git sha, config, system, projector).
    Results, if provided, go to ``response.npz`` for later re-rendering:

    - ``response``: ``(S, A, 3)`` intensity per pupil sample, FOV angle, color.
    - ``pupil_x_mm``, ``pupil_y_mm``: 1D axes for reshaping ``response`` to a
      2D pupil grid ``(ny, nx)``. ``S`` must equal ``len(y) * len(x)``.
    - ``scan_angles``: ``(n_fov_y, n_fov_x, 2)`` in radians. ``A`` must equal
      ``n_fov_y * n_fov_x``.
    - ``wavelengths_nm``: ``(3,)`` RGB trace wavelengths, in nanometers.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "git_sha": _git_sha(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "scan": _serialize_scan(scan),
        "projector": _serialize_projector(projector),
        "system": _serialize_system(system),
    }
    if extra is not None:
        manifest["extra"] = extra

    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    if response is not None:
        if pupil_x_mm is None or pupil_y_mm is None or scan_angles is None:
            raise ValueError(
                "response requires pupil_x_mm, pupil_y_mm, and scan_angles")
        arrays = {
            "response": np.asarray(response),
            "pupil_x_mm": np.asarray(pupil_x_mm),
            "pupil_y_mm": np.asarray(pupil_y_mm),
            "scan_angles": np.asarray(scan_angles),
        }
        if wavelengths_nm is not None:
            arrays["wavelengths_nm"] = np.asarray(wavelengths_nm)
        np.savez(run_dir / "response.npz", **arrays)

    return run_dir


def _serialize_fov_grid(fov_grid: FovGrid) -> dict:
    return {
        "x_fov": _round_sig(float(fov_grid.angles_grid[:, :, 0].max()
                                  - fov_grid.angles_grid[:, :, 0].min())),
        "y_fov": _round_sig(float(fov_grid.angles_grid[:, :, 1].max()
                                  - fov_grid.angles_grid[:, :, 1].min())),
        "num_x": int(fov_grid.num_x),
        "num_y": int(fov_grid.num_y),
    }


def _serialize_combiner_params(params) -> dict:
    return {
        "spacings": _to_list(params.spacings),
        "amplitudes": _to_list(params.amplitudes),
        "widths": _to_list(params.widths),
    }


def _serialize_merit_config(config) -> dict:
    return {
        "threshold_relative": _round_sig(float(config.threshold_relative)),
        "cap_relative": (_round_sig(float(config.cap_relative))
                         if config.cap_relative is not None else None),
        "sigmoid_steepness": _round_sig(float(config.sigmoid_steepness)),
        "soft_min_temperature": _round_sig(float(config.soft_min_temperature)),
        "shape_floor_epsilon": float(config.shape_floor_epsilon),
        "weight_shape": _round_sig(float(config.weight_shape)),
        "weight_coverage": _round_sig(float(config.weight_coverage)),
        "weight_warmup": _round_sig(float(config.weight_warmup)),
        "weight_cap": _round_sig(float(config.weight_cap)),
    }


def _serialize_merit_breakdown(breakdown: dict) -> dict:
    return {k: _round_sig(float(v)) for k, v in breakdown.items()}


def save_optimization_report(
    run_dir: Path | str,
    *,
    system: OpticalSystem,
    projectors: list[Projector],
    fov_grid: FovGrid,
    merit_config,
    optimizer_config: dict,
    param_bounds,
    initial_params,
    final_params,
    initial_breakdown: dict,
    final_breakdown: dict,
    loss_history: list[float],
    eyebox_config: dict,
) -> Path:
    """Write an optimization report to ``run_dir/optimization_report.json``.

    Reuses existing serializers for the system and projectors.  Captures
    everything needed to reproduce or compare an optimization run.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "git_sha": _git_sha(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "system": _serialize_system(system),
        "projectors": [_serialize_projector(p) for p in projectors],
        "fov_grid": _serialize_fov_grid(fov_grid),
        "eyebox": {
            "radius": _round_sig(float(eyebox_config["radius"])),
            "nx": int(eyebox_config["nx"]),
            "ny": int(eyebox_config["ny"]),
        },
        "merit_config": _serialize_merit_config(merit_config),
        "optimizer": optimizer_config,
        "param_bounds": {
            "spacing_min_mm": _round_sig(float(param_bounds.spacing_min_mm)),
            "spacing_max_mm": _round_sig(float(param_bounds.spacing_max_mm)),
            "amplitude_min": _round_sig(float(param_bounds.amplitude_min)),
            "amplitude_max": _round_sig(float(param_bounds.amplitude_max)),
            "width_min_nm": _round_sig(float(param_bounds.width_min_nm)),
            "width_max_nm": _round_sig(float(param_bounds.width_max_nm)),
            "chassis_usable_mm": _round_sig(float(param_bounds.chassis_usable_mm)),
        },
        "initial_params": _serialize_combiner_params(initial_params),
        "final_params": _serialize_combiner_params(final_params),
        "initial_merit": _serialize_merit_breakdown(initial_breakdown),
        "final_merit": _serialize_merit_breakdown(final_breakdown),
        "loss_history": [_round_sig(v) for v in loss_history],
    }

    path = run_dir / "optimization_report.json"
    path.write_text(json.dumps(report, indent=2))
    return path
