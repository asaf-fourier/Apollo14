"""Re-simulate Perseus with the REAL (TMM) coating reflectance and compare.

``design_perseus_mirror_coating.py`` wrote, per mirror, the **achieved**
reflectance of the physical thin-film stack Atlas found — the real ``R(λ, θ)``
its TMM predicts, which is *not* exactly the ideal target ``R(λ)`` the pupil
optimizer asked for. This script closes the loop: it reads that
``coating_design.json``, rebuilds the Perseus optical system with the real
coating reflectance in place of the ideal target, re-traces the eyebox, and
renders **the same pupil report** as ``optimize_pupil_perseus.py`` — so the
as-built performance can be compared against the as-designed intent.

To make the comparison apples-to-apples it reuses ``optimize_pupil_perseus``'s
exact trace configuration (projector, FOV grid, wavelength sampling, eyebox
grid, luminance weighting) — only the per-mirror reflectance differs. It builds
**two** systems from the JSON — one with the ideal target ``R(λ)``, one with
the real angle-averaged achieved ``R(λ)`` — renders a pupil report for each, and
prints an eyebox brightness diff so the deltas are quantified up front.

The Perseus tracer's mirrors are angle-independent, so the real reflectance is
reduced over the coating's angle samples (``ANGLE_REDUCTION``) to a single
``R(λ)`` per mirror.

No Atlas dependency — the TMM results are already baked into the JSON.

Run::

    python examples/analyze_perseus_coating.py
"""

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp

import examples.optimize_pupil_perseus as op  # reuse its exact trace config
from apollo14.elements.partial_mirror import PartialMirror
from apollo14.materials import air
from apollo14.perseus import build_perseus_geometry, spacings_for_count
from apollo14.system import OpticalSystem
from apollo14.trace import prepare_route
from apollo14.units import nm, mm
from helios.eyebox import trace_branch_over_fov
from helios.io import save_run, ScanConfig
from helios.merit import build_combiner_branch_routes
from helios.reports.pupil_report import render_pupil_report

# ── Input / config ───────────────────────────────────────────────────────────
# DESIGN_JSON = None → newest coating_design.json under examples/.
DESIGN_JSON: Path | None = None
ANGLE_REDUCTION = "mean"    # collapse achieved R(λ,θ) → R(λ): "mean" | "on_axis"
OUTPUT_ROOT = Path("examples/reports/analyze_perseus_coating")


def _latest_design_json() -> Path:
    matches = list(Path("examples").rglob(
        "design_perseus_mirror_coating/*/coating_design.json"))
    if not matches:
        raise FileNotFoundError(
            "No coating_design.json found. Run "
            "examples/design_perseus_mirror_coating.py first.")
    return max(matches, key=lambda path: path.stat().st_mtime)


# ── Read per-mirror reflectance from the design JSON ─────────────────────────

def _reflectance_at_trace(design: dict, kind: str) -> tuple[np.ndarray, int]:
    """``(num_mirrors, K)`` reflectance sampled on op.TRACE_WAVELENGTHS.

    ``kind`` selects the ideal target curve (``"target"``) or the real coating
    curve (``"achieved"``, reduced over angle by ``ANGLE_REDUCTION``).
    """
    num_mirrors = design["system"]["num_mirrors"]
    records = {mirror["index"]: mirror for mirror in design["mirrors"]}
    if set(records) != set(range(num_mirrors)):
        raise ValueError(
            f"JSON designed mirrors {sorted(records)} but the system has "
            f"{num_mirrors}; need every mirror to rebuild the full stack.")

    trace_wavelengths_nm = np.asarray(op.TRACE_WAVELENGTHS) / float(nm)
    curves = []
    for mirror_index in range(num_mirrors):
        record = records[mirror_index]
        if kind == "target":
            wavelengths_nm = np.asarray(
                record["target_reflectance"]["wavelengths_nm"])
            reflectance = np.asarray(record["target_reflectance"]["values"])
        else:
            achieved = record["result"]["achieved_reflectance"]
            wavelengths_nm = np.asarray(achieved["wavelengths_nm"])
            values = np.asarray(achieved["values"])           # (W, A)
            if ANGLE_REDUCTION == "on_axis":
                angles = np.asarray(achieved["angles_deg"])
                on_axis = design["system"]["angle_of_incidence_deg"]["on_axis"]
                reflectance = np.array([np.interp(on_axis, angles, row)
                                        for row in values])
            else:
                reflectance = values.mean(axis=1)             # angle-averaged
        curves.append(np.interp(trace_wavelengths_nm, wavelengths_nm, reflectance))
    return np.asarray(curves), num_mirrors


# ── Build a Perseus system with prescribed per-mirror reflectance ────────────

def _build_system(reflectance_per_mirror: np.ndarray, num_mirrors: int):
    geometry = build_perseus_geometry(spacings=spacings_for_count(num_mirrors))
    system = OpticalSystem(env_material=air)
    system.add(geometry.chassis)
    system.add(geometry.aperture)
    for mirror_index in range(num_mirrors):
        system.add(PartialMirror(
            name=f"mirror_{mirror_index}",
            position=geometry.mirror_positions[mirror_index],
            normal=geometry.mirror_normal.copy(),
            width=geometry.mirror_width, height=geometry.mirror_height,
            reflectance=jnp.asarray(reflectance_per_mirror[mirror_index]),
            wavelengths=op.TRACE_WAVELENGTHS,
        ))
    system.add(geometry.pupil)
    return system


# ── Eyebox response — op's trace loop, on a prescribed system ────────────────

def _spectral_response(system, num_mirrors: int) -> jnp.ndarray:
    """``(S, A, N)`` per-wavelength eyebox radiance — identical trace to op."""
    branch_routes = build_combiner_branch_routes(system, num_mirrors=num_mirrors)
    directions = op.FOV_GRID.flat_directions

    def trace_one_wavelength(_, wavelength):
        binned = jnp.zeros((directions.shape[0], op.EYEBOX_POINTS.shape[0]))
        for route in branch_routes:
            binned = binned + trace_branch_over_fov(
                prepare_route(route, wavelength), op.PROJECTOR, op.EYEBOX_POINTS,
                wavelength, directions, sigma=None, vmap_directions=True)
        windowed = op._window_mean(binned.T, op.SAMPLE_NY, op.SAMPLE_NX,
                                   op.KERNEL_SIZE_CELLS)
        return None, windowed

    _, responses = jax.lax.scan(trace_one_wavelength, None, op.TRACE_WAVELENGTHS)
    return jnp.transpose(responses, (1, 2, 0))


def _eyebox_brightness(response) -> jnp.ndarray:
    """Relative photometric brightness per eyebox cell, shaped (ny, nx)."""
    luminance_per_angle = jnp.sum(
        response * op.LUMINANCE_TRACE_WEIGHTS.reshape(1, 1, -1), axis=-1)
    relative = jnp.mean(luminance_per_angle, axis=-1) / op.INPUT_FLUX
    return relative.reshape(op.EYEBOX_NY, op.EYEBOX_NX)


# ── Report writing (op's save_run + render_pupil_report) ─────────────────────

def _save_eyebox_figure(path: Path, ideal, real, delta) -> None:
    """Side-by-side eyebox heatmaps: ideal target | real TMM | difference.

    Ideal and real share one brightness scale so the panels are directly
    comparable; the difference uses a diverging scale centered on zero.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    half_x = float(op.EYEBOX_HALF_X / mm)
    half_y = float(op.EYEBOX_HALF_Y / mm)
    extent = [-half_x, half_x, -half_y, half_y]
    brightness_max = float(max(ideal.max(), real.max()))
    brightness_min = float(min(ideal.min(), real.min()))
    delta_scale = float(np.abs(delta).max()) or 1e-12

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4))
    panels = [
        (axes[0], ideal, f"Ideal (target R)\nmean {ideal.mean():.2e}",
         "viridis", dict(vmin=brightness_min, vmax=brightness_max)),
        (axes[1], real, f"Real (TMM R)\nmean {real.mean():.2e}",
         "viridis", dict(vmin=brightness_min, vmax=brightness_max)),
        (axes[2], delta, f"Δ = real − ideal\nRMS {np.sqrt((delta ** 2).mean()):.2e}",
         "RdBu_r", dict(vmin=-delta_scale, vmax=delta_scale)),
    ]
    for axis, grid, title, cmap, scale in panels:
        image = axis.imshow(grid, origin="lower", extent=extent, aspect="equal",
                            cmap=cmap, **scale)
        axis.set_title(title, fontsize=10)
        axis.set_xlabel("eyebox x (mm)")
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    axes[0].set_ylabel("eyebox y (mm)")

    ratio = float(real.mean() / ideal.mean())
    fig.suptitle("Perseus eyebox brightness — as-designed vs as-built  "
                 f"(real / ideal = {ratio:.3f}, {(ratio - 1) * 100:+.1f}%)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def _write_report(run_dir: Path, system, response) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    pupil_x_mm = jnp.linspace(-op.EYEBOX_HALF_X + op._CELL_PITCH_X / 2,
                              op.EYEBOX_HALF_X - op._CELL_PITCH_X / 2, op.EYEBOX_NX)
    pupil_y_mm = jnp.linspace(-op.EYEBOX_HALF_Y + op._CELL_PITCH_Y / 2,
                              op.EYEBOX_HALF_Y - op._CELL_PITCH_Y / 2, op.EYEBOX_NY)
    scan_cfg = ScanConfig(
        base_direction=op.PROJECTOR_DIRECTION,
        x_fov=float(op.X_FOV), y_fov=float(op.Y_FOV),
        num_x=op.FOV_GRID.num_x, num_y=op.FOV_GRID.num_y)
    save_run(run_dir, system, op.PROJECTOR, scan_cfg, response=response,
             pupil_x_mm=pupil_x_mm, pupil_y_mm=pupil_y_mm,
             scan_angles=op.FOV_GRID.angles_grid,
             wavelengths_nm=op.TRACE_WAVELENGTHS / nm)
    return render_pupil_report(run_dir)


# ── Run ──────────────────────────────────────────────────────────────────────

def main():
    design_json = DESIGN_JSON or _latest_design_json()
    design = json.loads(Path(design_json).read_text())
    num_mirrors = design["system"]["num_mirrors"]

    target_reflectance, _ = _reflectance_at_trace(design, "target")
    achieved_reflectance, _ = _reflectance_at_trace(design, "achieved")

    print("── Perseus: ideal target vs real (TMM) coating ──")
    print(f"design JSON  : {design_json}")
    print(f"mirrors      : {num_mirrors}  (achieved R reduced over angle: {ANGLE_REDUCTION})")
    print(f"AOI range    : {design['system']['angle_of_incidence_deg']['min']:.1f}"
          f"–{design['system']['angle_of_incidence_deg']['max']:.1f}°")

    # Per-mirror band-mean reflectance — the headline "as-designed vs as-built".
    print(f"\n{'mirror':>6}  {'target R̄':>9}  {'real R̄':>8}  {'Δ (real−ideal)':>15}")
    for mirror_index in range(num_mirrors):
        target_mean = float(target_reflectance[mirror_index].mean())
        real_mean = float(achieved_reflectance[mirror_index].mean())
        print(f"{mirror_index:>6}  {target_mean:>9.4f}  {real_mean:>8.4f}  "
              f"{real_mean - target_mean:>+15.4f}")

    for i in range(8):
        target_reflectance[i] = target_reflectance[1]
    for i in range(8):
        achieved_reflectance[i] = achieved_reflectance[1]

    ideal_system = _build_system(target_reflectance, num_mirrors)
    real_system = _build_system(achieved_reflectance, num_mirrors)

    print("\nTracing eyebox (ideal target reflectance)...")
    ideal_response = _spectral_response(ideal_system, num_mirrors)
    print("Tracing eyebox (real TMM reflectance)...")
    real_response = _spectral_response(real_system, num_mirrors)

    ideal_brightness = np.asarray(_eyebox_brightness(ideal_response))
    real_brightness = np.asarray(_eyebox_brightness(real_response))
    delta = real_brightness - ideal_brightness

    def _stats(grid):
        return (f"mean={grid.mean():.5f}  std={grid.std():.5f}  "
                f"min={grid.min():.5f}  max={grid.max():.5f}")
    print("\n── Eyebox brightness (relative to input flux) ──")
    print(f"ideal (target R) : {_stats(ideal_brightness)}")
    print(f"real  (TMM R)    : {_stats(real_brightness)}")
    print(f"difference       : mean={delta.mean():+.5f}  "
          f"RMS={np.sqrt((delta ** 2).mean()):.5f}  "
          f"max|Δ|={np.abs(delta).max():.5f}")
    mean_ratio = real_brightness.mean() / ideal_brightness.mean()
    print(f"real/ideal mean brightness ratio: {mean_ratio:.3f} "
          f"({(mean_ratio - 1) * 100:+.1f}%)")

    print("\nΔ brightness map (real − ideal), per eyebox cell:")
    for row in delta:
        print("  " + "  ".join(f"{value:+.4f}" for value in row))

    run_dir = OUTPUT_ROOT / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    eyebox_figure = run_dir / "eyebox_ideal_vs_real.png"
    _save_eyebox_figure(eyebox_figure, ideal_brightness, real_brightness, delta)

    ideal_report = _write_report(run_dir / "ideal_target", ideal_system,
                                 ideal_response)
    real_report = _write_report(run_dir / "real_tmm", real_system, real_response)
    print(f"\nSaved:")
    print(f"  eyebox figure   : {eyebox_figure}")
    print(f"  ideal report    : {ideal_report}")
    print(f"  real  report    : {real_report}")


if __name__ == "__main__":
    main()
