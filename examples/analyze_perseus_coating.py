"""Analyze physical coatings on the exact saved Perseus system.

Replays saved surfaces, projector spectrum and FOV; retains angle dependence,
absorption and individual wafer recipes. Seeded physical film perturbations
are propagated through cached ray geometry into brightness/color statistics.

    uv run python examples/analyze_perseus_coating.py --design PATH/coating_design.json

The standalone coating_analysis.html includes nominal/ideal comparison,
manufacturing distributions, layer prescriptions, provenance and limitations.
"""

# Atlas must precede helios (its package initializer imports JAX).
# ruff: noqa: I001
import atlas

import argparse
import csv
import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from helios.coating_analysis import (
    build_geometry_cache,
    combine_paths,
    incidence_angles,
    interpolate_coating,
    load_context,
    response_metrics,
    validate_design,
)

# Import Atlas first, before the JAX-based tracer configures its backend.
from helios.coating_tolerance import CoatingModel, perturb_parameters
from helios.reports.coating_report import render_coating_report

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DESIGN_JSON: Path | None = None
OUTPUT_ROOT = PROJECT_ROOT / "examples/reports/analyze_perseus_coating"


def _latest_design_json():
    matches = sorted(
        (PROJECT_ROOT / "examples/reports/design_perseus_mirror_coating").glob(
            "*/coating_design.json"
        )
    )
    if not matches:
        raise FileNotFoundError(
            "No saved coating_design.json; run design_perseus_mirror_coating.py first"
        )
    return matches[-1]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--design", type=Path, default=DESIGN_JSON)
    parser.add_argument(
        "--source-report", type=Path, help="Override relocated source report directory"
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--samples",
        type=int,
        default=64,
        help="Manufacturing builds, default 64; zero = nominal only",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--thickness-fraction",
        type=float,
        help="Hard uniform ±fraction; default saved design value",
    )
    parser.add_argument(
        "--index-sigma", type=float, help="Gaussian recipe-index sigma; default saved design value"
    )
    parser.add_argument(
        "--thickness-correlation", choices=["independent", "wafer"], default="independent"
    )
    parser.add_argument(
        "--s-fraction",
        type=float,
        help="Incident s power fraction; default saved design polarization",
    )
    parser.add_argument(
        "--wavelengths", type=int, help="Override saved validation trace wavelength count"
    )
    parser.add_argument("--rays", type=int, nargs=2, metavar=("NX", "NY"))
    parser.add_argument("--angles", type=int, nargs=2, metavar=("NX", "NY"))
    parser.add_argument(
        "--coating-wavelengths",
        type=int,
        default=261,
        help="TMM grid across source band (default 1 nm for 420–680)",
    )
    parser.add_argument("--coating-angles", type=int, default=17)
    parser.add_argument(
        "--window-cells",
        type=int,
        default=3,
        help="Explicit moving-pupil window assumption; old reports do not save it",
    )
    parser.add_argument("--minimum-mean-ratio", type=float, default=0.95)
    parser.add_argument("--minimum-point-ratio", type=float, default=0.90)
    parser.add_argument(
        "--maximum-color-shift",
        type=float,
        default=0.01,
        help="Maximum CIE xy shift from ideal for diagnostic system pass",
    )
    args = parser.parse_args(argv)
    if args.samples < 0 or args.seed < 0:
        parser.error("samples and seed must be nonnegative")
    for name in ("wavelengths", "coating_wavelengths", "coating_angles"):
        value = getattr(args, name)
        if value is not None and value < 2:
            parser.error(f"{name} must be at least two")
    for name in ("rays", "angles"):
        value = getattr(args, name)
        if value and min(value) < 2:
            parser.error(f"{name} dimensions must be at least two")
    if args.window_cells < 1 or args.window_cells % 2 != 1:
        parser.error("window-cells must be positive and odd")
    for name in ("minimum_mean_ratio", "minimum_point_ratio", "maximum_color_shift"):
        if not np.isfinite(getattr(args, name)) or getattr(args, name) < 0:
            parser.error(f"{name} must be finite and nonnegative")
    return args


def _percentiles(values):
    values = np.asarray(values, dtype=float)
    return dict(
        zip(
            ["min", "p05", "p50", "p95", "max"],
            map(float, np.percentile(values, [0, 5, 50, 95, 100])),
            strict=True,
        )
    )


def _source_path(design, override):
    if override is not None:
        return override.resolve()
    path = Path(design["source_report"])
    return path if path.is_absolute() else PROJECT_ROOT / path


def _ratios(metrics, brightness, ideal_metrics, ideal_brightness):
    mean = ideal_metrics["mean_brightness"]
    if mean <= 0:
        raise ValueError("Ideal system is dark; relative manufacturing criteria are undefined")
    lit = ideal_brightness > 1e-12
    metrics["mean_brightness_ratio_to_ideal"] = metrics["mean_brightness"] / mean
    metrics["minimum_point_ratio_to_ideal"] = float(np.min(brightness[lit] / ideal_brightness[lit]))


def main(argv=None):
    args = parse_args(argv)
    design_path = (args.design or _latest_design_json()).resolve()
    design = json.loads(design_path.read_text())
    source_dir = _source_path(design, args.source_report)
    context = load_context(
        source_dir,
        wavelengths=args.wavelengths,
        rays=args.rays,
        angles=args.angles,
        window_cells=args.window_cells,
    )
    records = validate_design(design, context)
    source_glass = context.system.resolve(("chassis", "back"))._block_material.name
    coating_media = {record["result"]["incident_medium"] for record in records}
    media_match = coating_media == {source_glass}
    count = len(records)
    tolerance = design["design_params"]["tolerances"]
    fraction = (
        tolerance["thickness_plus_minus_fraction"]
        if args.thickness_fraction is None
        else args.thickness_fraction
    )
    sigma = tolerance["n_sigma"] if args.index_sigma is None else args.index_sigma
    if not np.isfinite(fraction) or not 0 <= fraction < 1 or not np.isfinite(sigma) or sigma < 0:
        raise ValueError("Invalid thickness fraction or index sigma")
    polarization = design["design_params"]["polarization"]
    default_polarization = {"s": 1.0, "p": 0.0, "unpolarized": 0.5}
    s_fraction = default_polarization[polarization] if args.s_fraction is None else args.s_fraction
    if not np.isfinite(s_fraction) or not 0 <= s_fraction <= 1:
        raise ValueError("s-fraction must be in [0,1]")
    output = args.output or OUTPUT_ROOT / datetime.now(UTC).strftime("%Y-%m-%d_%H-%M-%S_%f")
    output.mkdir(parents=True, exist_ok=False)
    print(f"Coating: {design_path}\nSaved system: {source_dir}\nOutput: {output}", flush=True)
    actual_angles = incidence_angles(context)
    wavelengths = np.linspace(
        context.wavelengths_nm[0], context.wavelengths_nm[-1], args.coating_wavelengths
    )
    # Include every saved validation angle as well as the actual dispersive FOV span.
    lower = min(
        actual_angles.min(),
        *(m["result"]["achieved_reflectance"]["angles_deg"][0] for m in records),
    )
    upper = max(
        actual_angles.max(),
        *(m["result"]["achieved_reflectance"]["angles_deg"][-1] for m in records),
    )
    angles = np.linspace(lower - 1e-4, upper + 1e-4, args.coating_angles)
    grid_r = np.empty((args.samples + 1, count, len(wavelengths), len(angles), 2), dtype=np.float32)
    grid_t = np.empty_like(grid_r)
    mirror_details = []
    draws = {}
    random_streams = np.random.SeedSequence(args.seed).spawn(count)
    for index, record in enumerate(records):
        print(
            f"Reconstructing wafer {index}; computing {args.samples} physical tolerance draws ...",
            flush=True,
        )
        model = CoatingModel(record, wavelengths, angles)
        grid_r[0, index], grid_t[0, index] = model.evaluate()
        rng = np.random.default_rng(random_streams[index])
        all_thicknesses = []
        all_indices = []
        outside = 0
        for sample in range(args.samples):
            thickness, index_values = perturb_parameters(
                model.thicknesses, model.indices, rng, fraction, sigma, args.thickness_correlation
            )
            outside += model.outside_index_support(index_values)
            grid_r[sample + 1, index], grid_t[sample + 1, index] = model.evaluate(
                thickness, index_values
            )
            all_thicknesses.append(thickness)
            all_indices.append(index_values)
        draws[f"mirror_{index}_thickness_nm"] = np.asarray(all_thicknesses).reshape(
            args.samples, len(model.thicknesses)
        )
        draws[f"mirror_{index}_recipe_indices"] = np.asarray(all_indices).reshape(
            args.samples, len(model.indices)
        )
        target = np.interp(
            wavelengths,
            record["target_reflectance"]["wavelengths_nm"],
            record["target_reflectance"]["values"],
        )
        errors = np.max(np.abs(grid_r[:, index, ..., 0] - target[:, None]), axis=(1, 2))
        mirror_details.append(
            {
                "mirror": index,
                "aoi_min_deg": float(actual_angles[index].min()),
                "aoi_max_deg": float(actual_angles[index].max()),
                "max_rs_error": float(errors[0]),
                "max_absorption": float(np.max(1 - grid_r[0, index] - grid_t[0, index])),
                "reconstruction_error": model.reconstruction_error,
                "tolerance_pass_fraction": float(
                    np.mean(errors[1:] <= tolerance["validation_max_rs_error"])
                )
                if args.samples
                else None,
                "index_draws_outside_support": outside,
                "nominal_indices_outside_support": model.outside_index_support(model.indices),
                "saved_tolerance_validation": record["result"].get("tolerance_validation"),
                "rs_max_error_percentiles": _percentiles(errors[1:]) if args.samples else None,
            }
        )
    print(
        "Tracing the saved geometry; recording finite upstream-mirror intersections ...", flush=True
    )
    cache = build_geometry_cache(
        context, count, progress=lambda message: print(message, flush=True)
    )
    cells = context.eyebox["nx"] * context.eyebox["ny"]

    def on_rays(grid):
        return np.stack(
            [
                interpolate_coating(
                    wavelengths, angles, grid[index], context.wavelengths_nm, actual_angles[index]
                )
                for index in range(count)
            ]
        )

    target_r = np.stack(
        [
            np.broadcast_to(
                np.interp(
                    context.wavelengths_nm,
                    m["target_reflectance"]["wavelengths_nm"],
                    m["target_reflectance"]["values"],
                )[None, :, None],
                (*actual_angles[index].shape, 2),
            )
            for index, m in enumerate(records)
        ]
    )
    ideal, ideal_contributions = combine_paths(cache, target_r, 1 - target_r, s_fraction)
    nominal, nominal_contributions = combine_paths(
        cache, on_rays(grid_r[0]), on_rays(grid_t[0]), s_fraction
    )
    ideal_metrics, ideal_brightness, ideal_color = response_metrics(ideal, context)
    nominal_metrics, nominal_brightness, nominal_color = response_metrics(
        nominal, context, reference=ideal
    )
    _ratios(nominal_metrics, nominal_brightness, ideal_metrics, ideal_brightness)
    sample_metrics = []
    brightness_draws = []
    color_draws = []
    system_pass = []
    for sample in range(args.samples):
        response, _ = combine_paths(
            cache,
            on_rays(grid_r[sample + 1]),
            on_rays(grid_t[sample + 1]),
            s_fraction,
            per_mirror=False,
        )
        metrics, brightness, color = response_metrics(response, context, reference=ideal)
        _ratios(metrics, brightness, ideal_metrics, ideal_brightness)
        passed = (
            media_match
            and metrics["mean_brightness_ratio_to_ideal"] >= args.minimum_mean_ratio
            and metrics["minimum_point_ratio_to_ideal"] >= args.minimum_point_ratio
            and metrics["maximum_delta_xy_from_ideal"] is not None
            and metrics["maximum_delta_xy_from_ideal"] <= args.maximum_color_shift
        )
        sample_metrics.append(metrics)
        brightness_draws.append(brightness)
        color_draws.append(color)
        system_pass.append(passed)
        if sample % 8 == 0 or sample == args.samples - 1:
            print(f"System tolerance build {sample + 1}/{args.samples}", flush=True)
    issues = []
    if not media_match:
        issues.append(
            f"Coating media {sorted(coating_media)} differ from saved chassis material {source_glass}. "
            "Results retain the saved ray geometry and are conditional; regenerate the source "
            "optimization with matching glass before assessing system pass rates."
        )
    if nominal_metrics["mean_brightness_ratio_to_ideal"] < args.minimum_mean_ratio:
        issues.append(
            f"Nominal mean brightness is {nominal_metrics['mean_brightness_ratio_to_ideal']:.1%} of ideal, below the diagnostic threshold."
        )
    if nominal_metrics["cell_fov_target_coverage"] < 1:
        issues.append(
            f"{1 - nominal_metrics['cell_fov_target_coverage']:.1%} of nominal cell/FOV samples fall below the source brightness target."
        )
    if (
        nominal_metrics["maximum_delta_xy_from_ideal"] is not None
        and nominal_metrics["maximum_delta_xy_from_ideal"] > args.maximum_color_shift
    ):
        issues.append("Nominal color shift from ideal exceeds the selected diagnostic limit.")
    if (
        nominal_metrics["maximum_d65_delta_xy"] is not None
        and nominal_metrics["maximum_d65_delta_xy"] > 0.01
    ):
        issues.append(
            f"Nominal maximum D65 color distance is {nominal_metrics['maximum_d65_delta_xy']:.4f} (report threshold 0.01); ideal baseline is {ideal_metrics['maximum_d65_delta_xy']:.4f}. This distinguishes baseline white-point error from coating-induced color shift."
        )
    for detail, record in zip(mirror_details, records, strict=True):
        index = detail["mirror"]
        saved = record["result"]["achieved_reflectance"]["angles_deg"]
        if detail["aoi_min_deg"] < saved[0] or detail["aoi_max_deg"] > saved[-1]:
            issues.append(
                f"M{index}: actual dispersive AOI extends beyond the saved coating angle grid; Atlas was reevaluated on expanded support."
            )
        if detail["max_rs_error"] > tolerance["validation_max_rs_error"]:
            issues.append(f"M{index}: nominal Rs error exceeds the saved coating criterion.")
        if detail["index_draws_outside_support"] or detail["nominal_indices_outside_support"]:
            issues.append(
                f"M{index}: recipe indices leave measured material support; Atlas clamps interpolation. Counts are reported; affected yield is conditional on that model."
            )
    if args.samples and np.mean(system_pass) < 1:
        issues.append(
            f"{sum(system_pass)}/{args.samples} simulated builds pass the selected system criteria."
        )
    if not issues:
        issues.append("No selected diagnostic thresholds were exceeded on this sampled grid.")
    configuration = {
        "samples": args.samples,
        "seed": args.seed,
        "thickness_fraction": fraction,
        "thickness_distribution": "bounded uniform",
        "thickness_correlation": args.thickness_correlation,
        "index_distribution": "independent Gaussian per film",
        "recipe_index_sigma": sigma,
        "minimum_mean_ratio_to_ideal": args.minimum_mean_ratio,
        "minimum_point_ratio_to_ideal": args.minimum_point_ratio,
        "maximum_delta_xy_from_ideal": args.maximum_color_shift,
        "d65_reporting_limit": 0.01,
        "coating_max_rs_error_limit": tolerance["validation_max_rs_error"],
        "criteria_status": "configurable diagnostic criteria, not manufacturing acceptance specifications",
    }
    scalar_keys = [key for key, value in nominal_metrics.items() if isinstance(value, float)]
    percentile_metrics = {
        key: _percentiles([row[key] for row in sample_metrics])
        for key in scalar_keys
        if args.samples and all(row.get(key) is not None for row in sample_metrics)
    }
    sampling = {
        "wavelengths": len(context.wavelengths_nm),
        "trace_band_nm": [float(context.wavelengths_nm[0]), float(context.wavelengths_nm[-1])],
        "coating_wavelengths": len(wavelengths),
        "coating_angles": len(angles),
        "projector_nx": context.projector.nx,
        "projector_ny": context.projector.ny,
        "fov_nx": context.fov.num_x,
        "fov_ny": context.fov.num_y,
        "eyebox_nx": context.eyebox["nx"],
        "eyebox_ny": context.eyebox["ny"],
        "window_cells": context.window_cells,
        "geometry_cache_paths": len(cache),
        "source_brightness_target": context.source["merit_config"]["target_relative"],
    }

    def sha(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    def git_sha(directory):
        return subprocess.check_output(
            ["git", "-C", str(directory), "rev-parse", "HEAD"], text=True
        ).strip()

    summary = {
        "provenance": {
            "coating_path": str(design_path),
            "source_report": str(source_dir),
            "coating_sha256": sha(design_path),
            "system_sha256": sha(source_dir / "optimization_report.json"),
            "source_git_sha": context.source["git_sha"],
            "analysis_git_sha": git_sha(PROJECT_ROOT),
            "atlas_git_sha": git_sha(Path(atlas.__file__).resolve().parents[1]),
            "analysis_code_sha256": {
                str(path.relative_to(PROJECT_ROOT)): sha(path)
                for path in [
                    Path(__file__),
                    PROJECT_ROOT / "helios/coating_analysis.py",
                    PROJECT_ROOT / "helios/coating_tolerance.py",
                    PROJECT_ROOT / "helios/reports/coating_report.py",
                ]
            },
            "created_utc": datetime.now(UTC).isoformat(),
            "s_polarization_fraction": s_fraction,
            "sampling": sampling,
        },
        "sampling": sampling,
        "system_geometry": [
            {
                key: element.get(key)
                for key in (
                    "name",
                    "type",
                    "position",
                    "normal",
                    "width",
                    "height",
                    "inner_width",
                    "inner_height",
                    "material",
                )
            }
            for element in context.source["system"]["elements"]
        ],
        "ideal": ideal_metrics,
        "nominal": nominal_metrics,
        "mirrors": mirror_details,
        "issues": issues,
        "tolerance": {
            "configuration": configuration,
            "samples": args.samples,
            "percentiles": percentile_metrics,
            "system_pass_fraction": float(np.mean(system_pass)) if args.samples else None,
        },
        "limitations": [
            "Sequential primary reflection routes only; ghost/multiple-reflection paths, scatter, roughness and coating phase shifts are not modeled.",
            "Geometry is restored from serialized source surfaces (six-significant-digit precision); source glass dispersion uses the local material catalog.",
            "The moving-window size was not stored in old source reports. It is explicit here; default 3 × 3 cells follows the optimizer convention.",
            "s/p channels are preserved separately for parallel mirrors; polarization-basis rotation across differing incidence planes is not modeled.",
            "Only film thickness and recipe-index errors are perturbed. Geometry, alignment, wafer wedge, temperature, bonding and substrate uncertainties are fixed.",
            "Coating values are interpolated from the reported TMM spectral/angular grid without extrapolation. Narrow features and spatial/FOV sampling need convergence checks.",
            "Gaussian index draws outside measured recipe support use Atlas endpoint clamping; counts are reported and affected predictions are conditional.",
            "A finite Monte Carlo sample is not a manufacturing-yield certification. System diagnostic thresholds are configurable.",
            "Ideal reference uses each saved target R and lossless T=1−R. Nominal uses physical Rs/Rp/Ts/Tp including absorption.",
            "Fixed scalar chassis AR losses are inherited from the source; bulk-glass absorption and polarization-dependent chassis Fresnel losses are not added.",
        ],
    }
    box = context.eyebox
    pupil_x = np.linspace(
        -box["half_x"] + box["half_x"] / box["nx"],
        box["half_x"] - box["half_x"] / box["nx"],
        box["nx"],
    )
    pupil_y = np.linspace(
        -box["half_y"] + box["half_y"] / box["ny"],
        box["half_y"] - box["half_y"] / box["ny"],
        box["ny"],
    )
    arrays = {
        "luminance_weights": context.luminance_weights,
        "input_flux": np.asarray(context.input_flux),
        "ideal_response": ideal[:cells],
        "nominal_response": nominal[:cells],
        "ideal_brightness": ideal_brightness,
        "nominal_brightness": nominal_brightness,
        "ideal_color": ideal_color,
        "nominal_color": nominal_color,
        "ideal_mirror_contributions": ideal_contributions,
        "nominal_mirror_contributions": nominal_contributions,
        "actual_angles_deg": actual_angles,
        "wavelengths_nm": context.wavelengths_nm,
        "scan_angles": np.asarray(context.fov.angles_grid),
        "pupil_x_mm": pupil_x,
        "pupil_y_mm": pupil_y,
        "coating_wavelengths_nm": wavelengths,
        "coating_angles_deg": angles,
        "nominal_grid_r": grid_r[0],
        "nominal_grid_t": grid_t[0],
        "tolerance_brightness": np.asarray(brightness_draws),
        "tolerance_color": np.asarray(color_draws),
        "tolerance_metrics_mean_brightness": np.asarray(
            [m["mean_brightness"] for m in sample_metrics]
        ),
    }
    np.savez_compressed(output / "analysis_arrays.npz", **arrays)
    np.savez_compressed(output / "tolerance_parameters.npz", **draws)
    (output / "analysis.json").write_text(json.dumps(summary, indent=2, allow_nan=False))
    (output / "coating_design.json").write_bytes(design_path.read_bytes())
    (output / "source_optimization_report.json").write_bytes(
        (source_dir / "optimization_report.json").read_bytes()
    )
    with (output / "tolerance_samples.csv").open("w") as handle:
        keys = list(nominal_metrics)
        writer = csv.DictWriter(handle, fieldnames=["sample", "system_pass", *keys])
        writer.writeheader()
        for index, metrics in enumerate(sample_metrics):
            writer.writerow({"sample": index, "system_pass": bool(system_pass[index]), **metrics})
    report = render_coating_report(output, summary, arrays, design)
    print(
        f"Nominal / ideal mean brightness: {nominal_metrics['mean_brightness_ratio_to_ideal']:.5f}\nReport: {report}",
        flush=True,
    )
    return output


if __name__ == "__main__":
    main()
