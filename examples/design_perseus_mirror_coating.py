"""Design physical thin-film coatings for the Perseus mirror stack with **Atlas**.

This is the bridge between the two projects. ``optimize_pupil_perseus.py``
produces, per mirror, a *target reflectance curve* ``R(λ)`` — the fraction of
light each partial mirror should pick off, per wavelength. That curve is an
abstract spec; it says nothing about how to build the mirror. **Atlas**
(``/PycharmProjects/Atlas``) is the thin-film designer that turns such a spec
into a real multilayer stack (materials + thicknesses) via differentiable TMM.

This example takes the output of ``optimize_pupil_perseus.py`` and, **for every
mirror in the stack**, hands Atlas everything it needs to find its coating:

- **target reflectance** ``R(λ)`` — read straight from the optimizer's saved
  report (each mirror element stores its reflectance sampled over wavelength);
- **angle(s) of incidence** — the beam refracts into the ophthalmic glass at the
  combiner's entry face, so the mirror is hit at an in-glass AOI; we sweep the
  FOV to get the angle *range* the coating must hold. All mirrors share one
  normal, so the AOI is common to the stack — only ``R(λ)`` differs per mirror;
- **projector spectrum** — the PlayNitride panel's emission, used to weight the
  fit toward the wavelengths that actually carry light;
- **substrate/medium** — the mirror is embedded in index-matched ophthalmic
  glass (``Materials.AGC_M074``), so the coating sits glass-on-both-sides.

Every mirror is designed exactly as in Atlas's ``two_step_basin_hopping.py``
(basin-hopping global search → L-BFGS polish), and the whole run is written to a
single consolidated JSON holding the **system + mirror state**, the **design
parameters**, and the **per-mirror results** (coating recipes + achieved R).

Prerequisite: run ``optimize_pupil_perseus.py`` first (this reads its latest
report). ``atlas`` must be importable — added as an editable path dependency
(``uv add --editable ../Atlas``).

Run::

    python examples/design_perseus_mirror_coating.py
"""

# Import atlas BEFORE jax so its XLA multi-core flag takes effect (see
# atlas/__init__.py). apollo14 (which imports jax) is imported after.
from atlas import OpticalDesigner, Materials, Layer, Target, example_output_dir

import json
import time
from datetime import datetime, UTC
from pathlib import Path

import numpy as np
import jax.numpy as jnp

from apollo14.units import nm, mm
from apollo14.materials import agc_m074
from apollo14.geometry import snell_refract
from apollo14.perseus import (
    PERSEUS_PROJECTOR_DIRECTION,
    PERSEUS_LIGHT_POSITION,
    PERSEUS_BEAM_WIDTH,
    PERSEUS_BEAM_HEIGHT,
    PERSEUS_FOV_AROUND_X,
    PERSEUS_FOV_AROUND_PROJECTOR_Y,
    build_perseus_geometry,
    spacings_for_count,
)
from apollo14.projector import PlayNitrideLed, FovGrid

# ── Which optimizer output, which mirrors ───────────────────────────────────
# REPORT_DIR = None → auto-pick the newest optimize_pupil_perseus run.
# MIRROR_INDICES = None → design every mirror in the stack; or a list for a subset.
OPTIMIZE_RUNS_ROOT = Path("examples/reports/optimize_pupil_perseus")
REPORT_DIR: Path | None = None
MIRROR_INDICES: list[int] | None = None

# ── Coating design knobs (fed to Atlas) ─────────────────────────────────────
REFERENCE_WAVELENGTH_NM = 550.0     # glass index / QWOT seed reference
POLARIZATION = "both"               # combiner light is effectively unpolarized
NUM_WAVELENGTHS = 120               # target sample points across the band
NUM_ANGLES = 5                      # AOI samples across the swept range

# Alternating high/low-index films (DBR-style contrast) in AGC M-074 glass.
NUM_SEED_FILMS = 9
THICKNESS_BOUNDS_NM = (20.0, 300.0)
TIO2_N_BOUNDS = (2.0, 2.525)        # PLD_TiO2 tunable index range
AL2O3_N_BOUNDS = (1.47, 1.65)       # PLD_Al2O3 tunable index range
SEED_N_HIGH = 2.50                  # QWOT seed indices (max Δn)
SEED_N_LOW = 1.47
THICKNESS_TOLERANCE_NM = 1
N_TOLERANCE = 0.005

# Optimizer budget PER MIRROR. Modest defaults so a full stack runs in minutes;
# bump for production recipes (BH max_iterations→100+, local_maxiter→500).
BH_MAX_ITERATIONS = 300
BH_LOCAL_MAXITER = 400
BH_STEPSIZE = 0.2
BH_TEMPERATURE = 0.01
POLISH_MAX_ITERATIONS = 1500

SAVE_PER_MIRROR_PLOTS = True
OUTPUT_ROOT = "examples/reports"


# ── Load the optimizer output ────────────────────────────────────────────────

def _latest_report_dir() -> Path:
    reports = sorted(OPTIMIZE_RUNS_ROOT.glob("*/optimization_report.json"))
    if not reports:
        raise FileNotFoundError(
            f"No optimization_report.json under {OPTIMIZE_RUNS_ROOT}. "
            f"Run examples/optimize_pupil_perseus.py first.")
    return reports[-1].parent


def load_report(report_dir: Path):
    """Return (mirrors, num_mirrors, curve_mode, size_wh_mm, mirror_normal,
    source_optimizer) parsed from an optimize_pupil_perseus report.

    ``mirrors`` is a list of ``(index, wavelengths_nm, reflectance)`` — one per
    PartialMirror. Size and normal are common to the stack (all mirrors share
    the same geometry), so they are returned once from the first mirror.
    """
    report = json.loads((report_dir / "optimization_report.json").read_text())
    elements = [e for e in report["system"]["elements"]
                if e["type"] == "PartialMirror"]
    elements.sort(key=lambda e: int(e["name"].split("_")[1]))

    mirrors = []
    for mirror_index, element in enumerate(elements):
        wavelengths_nm = np.asarray(element["wavelengths"]) / float(nm)
        reflectance = np.asarray(element["reflectance"])
        order = np.argsort(wavelengths_nm)
        mirrors.append((mirror_index, wavelengths_nm[order], reflectance[order]))

    size_wh_mm = (float(elements[0]["width"]) / mm,
                  float(elements[0]["height"]) / mm)
    return (mirrors, len(elements), report["optimizer"].get("curve_mode", "?"),
            size_wh_mm, np.asarray(elements[0]["normal"]), report["optimizer"])


# ── Angle of incidence in the glass ──────────────────────────────────────────

def _incidence_angle_deg(beam_direction, entry_normal, mirror_normal,
                         glass_index) -> float:
    """AOI on the mirror after the beam refracts into the glass, in degrees."""
    beam = np.asarray(beam_direction)
    beam = beam / np.linalg.norm(beam)
    face_normal = np.asarray(entry_normal)
    face_normal = face_normal / np.linalg.norm(face_normal)
    if np.dot(beam, face_normal) > 0:      # snell_refract wants it against the ray
        face_normal = -face_normal
    in_glass, _tir = snell_refract(jnp.asarray(beam), jnp.asarray(face_normal),
                                   1.0, glass_index)
    in_glass = np.asarray(in_glass)
    mirror_unit = np.asarray(mirror_normal)
    mirror_unit = mirror_unit / np.linalg.norm(mirror_unit)
    cos_incidence = abs(float(np.dot(in_glass, mirror_unit)))
    return float(np.degrees(np.arccos(np.clip(cos_incidence, 0.0, 1.0))))


def incidence_angle_range(mirror_normal, num_mirrors, glass_index):
    """(min, max, on_axis) AOI in degrees, swept over the full FOV."""
    geometry = build_perseus_geometry(spacings=spacings_for_count(num_mirrors))
    entry_normal = geometry.aperture.normal   # entry face ‖ the aperture
    fov = FovGrid(PERSEUS_PROJECTOR_DIRECTION, PERSEUS_FOV_AROUND_X,
                  PERSEUS_FOV_AROUND_PROJECTOR_Y, num_x=5, num_y=5)
    angles = [_incidence_angle_deg(direction, entry_normal, mirror_normal,
                                   glass_index)
              for direction in fov.flat_directions]
    on_axis = _incidence_angle_deg(PERSEUS_PROJECTOR_DIRECTION, entry_normal,
                                   mirror_normal, glass_index)
    return min(angles), max(angles), on_axis, sorted(angles)


# ── Projector spectrum → per-wavelength fit weights ──────────────────────────

def projector_weights(wavelength_grid_nm) -> list[float]:
    """Weight the fit toward wavelengths the panel actually emits.

    ``0.2 + 0.8·(radiance / peak)`` — every wavelength keeps a ≥0.2 floor so the
    coating stays sane out-of-peak, while the RGB emission bands dominate.
    """
    projector = PlayNitrideLed.create_broadband(
        position=PERSEUS_LIGHT_POSITION, direction=PERSEUS_PROJECTOR_DIRECTION,
        beam_width=PERSEUS_BEAM_WIDTH, beam_height=PERSEUS_BEAM_HEIGHT,
        nx=1, ny=1)
    spec_wavelengths, spec_radiance = projector.spectrum
    spec_wavelengths_nm = np.asarray(spec_wavelengths) / float(nm)
    spec_radiance = np.asarray(spec_radiance)
    radiance_on_grid = np.interp(wavelength_grid_nm, spec_wavelengths_nm,
                                 spec_radiance)
    return (0.2 + 0.8 * radiance_on_grid / radiance_on_grid.max()).tolist()


# ── QWOT seed thickness ──────────────────────────────────────────────────────

def _qwot_thickness_nm(center_wavelength_nm, film_index, medium_index,
                       incidence_deg) -> float:
    """Quarter-wave optical thickness at the in-medium incidence angle."""
    sin_film = medium_index * np.sin(np.radians(incidence_deg)) / film_index
    cos_film = np.sqrt(max(0.0, 1.0 - sin_film ** 2))
    return center_wavelength_nm / (4.0 * film_index * cos_film)


def _sample_curve_for_print(wavelengths_nm, reflectance, num=9):
    """A few ``(wavelength_nm, R)`` probes across the band, for readable print."""
    probes = np.linspace(wavelengths_nm.min(), wavelengths_nm.max(), num)
    return [(float(wl), float(np.interp(wl, wavelengths_nm, reflectance)))
            for wl in probes]


def _seed_layers(glass, seed_thickness_high, seed_thickness_low) -> list:
    """A fresh alternating PLD_TiO2 / PLD_Al2O3 seed stack in ``glass``."""
    layers = [Layer(material=glass)]                       # medium = glass
    for film_idx in range(NUM_SEED_FILMS):
        is_high = film_idx % 2 == 0
        layers.append(Layer(
            material=Materials.PLD_TiO2 if is_high else Materials.PLD_Al2O3,
            thickness=seed_thickness_high if is_high else seed_thickness_low,
            refractive_index=SEED_N_HIGH if is_high else SEED_N_LOW,
            vary_thickness=True, vary_n=True,
            thickness_bounds=THICKNESS_BOUNDS_NM,
            thickness_tolerance=THICKNESS_TOLERANCE_NM,
            n_tolerance=N_TOLERANCE,
            n_bounds=TIO2_N_BOUNDS if is_high else AL2O3_N_BOUNDS,
        ))
    layers.append(Layer(material=glass))                   # substrate = glass
    return layers

def _seed_moveon_layers(moveon, seed_thickness_high, seed_thickness_low) -> list:
    """moveon materials only."""
    layers = [Layer(material=moveon.MR10)]
    stack_materials = [moveon.SiO2, moveon.Ti3O5, moveon.SiO2, moveon.TiO2, moveon.Al2O3, moveon.TiO2, moveon.SiO2, moveon.Ti3O5, moveon.SiO2]
    for i, material in enumerate(stack_materials):
        is_high = i % 2 == 1
        layers.append(Layer(
            material=material,
            thickness=seed_thickness_high if is_high else seed_thickness_low,
            vary_thickness=True,
            thickness_bounds=THICKNESS_BOUNDS_NM,
            thickness_tolerance=THICKNESS_TOLERANCE_NM,
        ))
    layers.append(Layer(material=moveon.MR10))

    return layers



# ── Design one mirror (basin hopping → L-BFGS polish) ────────────────────────

def design_mirror(mirror_index, target_wavelengths_nm, target_reflectance,
                  aoi_range, weights, glass, seed_thickness_high,
                  seed_thickness_low):
    """Run the two-step Atlas design for one mirror; return its result."""
    target = Target(
        wavelengths=(float(target_wavelengths_nm.min()),
                     float(target_wavelengths_nm.max())),
        angles=aoi_range,
        reflectance=lambda wl: float(np.interp(wl, target_wavelengths_nm,
                                               target_reflectance)),
        polarization=POLARIZATION,
        num_wavelengths=NUM_WAVELENGTHS,
        num_angles=NUM_ANGLES,
        weights=weights,
    )

    designer = OpticalDesigner(
        # layers=_seed_layers(glass, seed_thickness_high, seed_thickness_low),
        layers=_seed_moveon_layers(Materials.moveon, seed_thickness_high, seed_thickness_low),
        target=target)
    hop_state = {"hop": 0, "best": float("inf")}

    def hop_progress(x, merit, accept):
        hop_state["hop"] += 1
        merit = float(merit)
        marker = "★" if merit < hop_state["best"] else " "
        hop_state["best"] = min(hop_state["best"], merit)
        print(f"    mirror {mirror_index}: hop {hop_state['hop']:>3}/{BH_MAX_ITERATIONS} "
              f" merit={merit:.4e} best={hop_state['best']:.4e} {marker}", flush=True)

    result_bh = designer.optimize(
        method="basin_hopping", max_iterations=BH_MAX_ITERATIONS, robust=False,
        stepsize=BH_STEPSIZE, temperature=BH_TEMPERATURE,
        local_maxiter=BH_LOCAL_MAXITER, callback=hop_progress)

    refined = [result_bh.layers[0]] + result_bh.film_layers + [result_bh.layers[-1]]
    return OpticalDesigner(layers=refined, target=target).optimize(
        method="lbfgs", max_iterations=POLISH_MAX_ITERATIONS)


# ── Serialize one Atlas result ───────────────────────────────────────────────

def _layer_to_dict(layer) -> dict:
    """Serializable view of one film, matching Atlas's own layer schema.

    ``refractive_index`` is the *design variable* — a number for tunable films,
    ``None`` for tabulated ones like the MoveOn set. ``n``/``k`` are the
    resolved physical index from ``Layer.index_at``, which reads the tunable
    parameter or the material's measured dispersion as appropriate.
    """
    nk = layer.index_at(REFERENCE_WAVELENGTH_NM)
    return {
        "material": layer.material.name,
        "thickness_nm": float(layer.thickness),
        "refractive_index": (None if layer.refractive_index is None
                             else float(layer.refractive_index)),
        "n": float(nk.real),
        "k": float(nk.imag),
        "is_tunable": bool(layer.material.is_tunable),
    }


def _result_to_dict(result) -> dict:
    return {
        "merit": float(result.merit),
        "nominal_merit": (None if result.nominal_merit is None
                          else float(result.nominal_merit)),
        "converged": bool(result.converged),
        "num_films": len(result.film_layers),
        "layers": [_layer_to_dict(layer) for layer in result.film_layers],
        "achieved_reflectance": {
            "wavelengths_nm": np.asarray(result.wavelengths_nm).tolist(),
            "angles_deg": np.asarray(result.angles_deg).tolist(),
            "values": np.asarray(result.reflectance).tolist(),   # (W, A)
        },
    }


# ── Run ──────────────────────────────────────────────────────────────────────

def main():
    report_dir = REPORT_DIR or _latest_report_dir()
    (mirrors, num_mirrors, curve_mode, size_wh_mm, mirror_normal,
     source_optimizer) = load_report(report_dir)

    indices = (list(range(num_mirrors)) if MIRROR_INDICES is None
               else MIRROR_INDICES)

    # Geometry / spectrum shared by the whole stack (mirrors share one normal).
    glass_index = float(jnp.interp(REFERENCE_WAVELENGTH_NM * nm,
                                   agc_m074.wavelengths, agc_m074.n_values))
    aoi_min, aoi_max, aoi_on_axis, fov_ray_angles = incidence_angle_range(
        mirror_normal, num_mirrors, glass_index)
    aoi_design = 0.5 * (aoi_min + aoi_max)
    seed_thickness_high = _qwot_thickness_nm(
        REFERENCE_WAVELENGTH_NM, SEED_N_HIGH, glass_index, aoi_design)
    seed_thickness_low = _qwot_thickness_nm(
        REFERENCE_WAVELENGTH_NM, SEED_N_LOW, glass_index, aoi_design)

    band = (float(mirrors[0][1].min()), float(mirrors[0][1].max()))
    weights = projector_weights(np.linspace(*band, NUM_WAVELENGTHS))
    glass = Materials.AGC_M074

    print("── Perseus mirror stack → Atlas coatings ──")
    print(f"source report : {report_dir}  (pupil-optimizer mode: {curve_mode})")
    print(f"designing     : {len(indices)} of {num_mirrors} mirrors  "
          f"(size {size_wh_mm[0]:.1f} × {size_wh_mm[1]:.2f} mm each)")
    print(f"glass         : {glass.name}, n({REFERENCE_WAVELENGTH_NM:.0f}nm)={glass_index:.4f} "
          f"(medium = substrate; mirror index-matched in glass)")
    print(f"AOI in glass  : on-axis {aoi_on_axis:.1f}°, range {aoi_min:.1f}°–{aoi_max:.1f}°")
    angle_grid = np.linspace(aoi_min, aoi_max, NUM_ANGLES)
    print("  ray AOI over FOV (deg): "
          + ", ".join(f"{a:.2f}" for a in fov_ray_angles))
    print(f"  Atlas design angles ({NUM_ANGLES} samples, deg): "
          + ", ".join(f"{a:.2f}" for a in angle_grid))
    print(f"seed stack    : {NUM_SEED_FILMS} films ({NUM_SEED_FILMS // 2} H/L pairs), "
          f"QWOT@{aoi_design:.0f}° H={seed_thickness_high:.0f}nm L={seed_thickness_low:.0f}nm")
    print(f"budget/mirror : basin_hopping {BH_MAX_ITERATIONS}×{BH_LOCAL_MAXITER}, "
          f"lbfgs polish {POLISH_MAX_ITERATIONS}\n")

    out_dir = Path(example_output_dir(__file__, root=OUTPUT_ROOT))

    mirror_records = []
    run_start = time.perf_counter()
    for design_position, mirror_index in enumerate(indices):
        _, wavelengths_nm, reflectance = mirrors[mirror_index]
        print(f"[{design_position + 1}/{len(indices)}] mirror {mirror_index}: "
              f"target R {reflectance.min():.3f}..{reflectance.max():.3f}")
        print("    target R(λ): " + "  ".join(
            f"{int(wl)}nm={r:.3f}"
            for wl, r in _sample_curve_for_print(wavelengths_nm, reflectance)))
        result = design_mirror(
            mirror_index, wavelengths_nm, reflectance, (aoi_min, aoi_max),
            weights, glass, seed_thickness_high, seed_thickness_low)
        print(f"    → merit {result.merit:.4e}, {len(result.film_layers)} films")

        if SAVE_PER_MIRROR_PLOTS:
            result.plot_per_angle(
                filename=str(out_dir / f"coating_mirror_{mirror_index}.png"),
                config_text=(f"Perseus mirror #{mirror_index}/{num_mirrors}\n"
                             f"AOI {aoi_min:.1f}–{aoi_max:.1f}°  merit {result.merit:.3e}"))

        mirror_records.append({
            "index": mirror_index,
            "target_reflectance": {
                "wavelengths_nm": wavelengths_nm.tolist(),
                "values": reflectance.tolist(),
            },
            "result": _result_to_dict(result),
        })

    elapsed = time.perf_counter() - run_start

    design = {
        "generated_utc": datetime.now(UTC).isoformat(),
        "source_report": str(report_dir),
        "wall_time_s": round(elapsed, 1),
        "system": {
            "num_mirrors": num_mirrors,
            "designed_mirrors": indices,
            "pupil_optimizer_mode": curve_mode,
            "mirror_size_mm": {"width": size_wh_mm[0], "height": size_wh_mm[1]},
            "mirror_normal": np.asarray(mirror_normal).tolist(),
            "glass": {
                "name": glass.name,
                "reference_wavelength_nm": REFERENCE_WAVELENGTH_NM,
                "index": glass_index,
            },
            "angle_of_incidence_deg": {
                "on_axis": aoi_on_axis, "min": aoi_min, "max": aoi_max,
            },
            "wavelength_range_nm": list(band),
            "projector": {
                "name": "PlayNitride broadband white",
                "position_mm": (np.asarray(PERSEUS_LIGHT_POSITION) / mm).tolist(),
                "direction": np.asarray(PERSEUS_PROJECTOR_DIRECTION).tolist(),
            },
            "source_pupil_optimizer": source_optimizer,
        },
        "design_params": {
            "polarization": POLARIZATION,
            "num_wavelengths": NUM_WAVELENGTHS,
            "num_angles": NUM_ANGLES,
            "fit_weights": "0.2 + 0.8·radiance/peak (projector-spectrum weighted)",
            "materials": {
                "medium_substrate": glass.name,
                "high_index": "PLD_TiO2", "high_n_bounds": list(TIO2_N_BOUNDS),
                "low_index": "PLD_Al2O3", "low_n_bounds": list(AL2O3_N_BOUNDS),
            },
            "seed": {
                "num_films": NUM_SEED_FILMS,
                "n_high": SEED_N_HIGH, "n_low": SEED_N_LOW,
                "thickness_high_nm": seed_thickness_high,
                "thickness_low_nm": seed_thickness_low,
                "design_aoi_deg": aoi_design,
            },
            "thickness_bounds_nm": list(THICKNESS_BOUNDS_NM),
            "tolerances": {"thickness_nm": THICKNESS_TOLERANCE_NM, "n": N_TOLERANCE},
            "basin_hopping": {
                "max_iterations": BH_MAX_ITERATIONS,
                "local_maxiter": BH_LOCAL_MAXITER,
                "stepsize": BH_STEPSIZE, "temperature": BH_TEMPERATURE,
            },
            "polish": {"method": "lbfgs", "max_iterations": POLISH_MAX_ITERATIONS},
        },
        "mirrors": mirror_records,
    }

    design_path = out_dir / "coating_design.json"
    design_path.write_text(json.dumps(design, indent=2))

    print(f"\n── Done ({elapsed:.0f}s) ──")
    print(f"{'mirror':>6}  {'films':>5}  {'merit':>11}  target R")
    for record in mirror_records:
        target_values = record["target_reflectance"]["values"]
        print(f"{record['index']:>6}  {record['result']['num_films']:>5}  "
              f"{record['result']['merit']:>11.3e}  "
              f"{min(target_values):.3f}..{max(target_values):.3f}")
    print(f"\nSaved consolidated design: {design_path}")
    if SAVE_PER_MIRROR_PLOTS:
        print(f"Per-mirror plots:          {out_dir}/coating_mirror_*.png")


if __name__ == "__main__":
    main()
