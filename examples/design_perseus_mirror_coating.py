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
  glass (``GLASS``, N-BASF2), so the coating sits glass-on-both-sides;
- **angular suppression** — besides matching ``R(λ)`` for s light at the
  display angles, each coating keeps Rp low there and keeps unpolarized
  reflection low at ``SUPPRESSION_REGIONS`` (10–36° and 60–75° by default: the
  angles at which ambient light reflects into TIR ghosts, from
  ``visualize_perseus_sun.py``), via an Atlas ``MultiTarget``.

Each mirror is designed in stages: basin hopping on the display alone, then
warm-started basin hopping through ``SUPPRESSION_WEIGHT_RAMP`` up to
``SUPPRESSION_WEIGHT``, then a robust L-BFGS polish under the ±1% thickness
tolerance. Each design then gets:

- an **angular report** — R(λ, θ) over 0–85° for s, p and unpolarized light,
  display Rs per LED color, worst / mean reflectance per suppression region
  (and where it sits), the two-reflection ghost product, and the unconstrained
  angles below the first region; printed, saved in the JSON, and plotted;
- a **tolerance analysis** — scrambled-Sobol Monte Carlo of bounded ±1%
  thickness and Gaussian index errors, scoring display Rs (yield) and the
  worst suppressed reflectance (percentiles).

The whole run is written to a single consolidated JSON holding the **system +
mirror state**, the **design parameters**, and the **per-mirror results**
(coating recipes, deposition order, achieved R, angular report, tolerance), plus
one interactive HTML per mirror (display fit, angular map, angular profile).

Prerequisite: run ``optimize_pupil_perseus.py`` first (this reads its latest
report). ``atlas`` must be importable — added as an editable path dependency
(``uv add --editable ../Atlas``).

Run::

    python examples/design_perseus_mirror_coating.py
"""

# Atlas must precede JAX here, so this intentionally differs from isort order.
# ruff: noqa: I001
import json
import time
from datetime import UTC, datetime
from pathlib import Path

# Import atlas BEFORE jax so its XLA multi-core flag takes effect (see
# atlas/__init__.py). apollo14 (which imports jax) is imported after.
from atlas import Layer, Materials, MultiTarget, OpticalDesigner, Target, example_output_dir
from atlas.core.stack import Stack
from atlas.core.tmm import _precompute_fixed_nk_data, compute_rt_polarized
import jax
import jax.numpy as jnp
import numpy as np
import plotly.graph_objects as go
from plotly.colors import sample_colorscale
from plotly.subplots import make_subplots
from scipy.stats import norm, qmc

from apollo14.geometry import snell_refract
from apollo14.materials import air, refractive_index
from apollo14.perseus import (
    PERSEUS_BEAM_HEIGHT,
    PERSEUS_BEAM_WIDTH,
    PERSEUS_FOV_AROUND_PROJECTOR_Y,
    PERSEUS_FOV_AROUND_X,
    PERSEUS_LIGHT_POSITION,
    PERSEUS_PROJECTOR_DIRECTION,
    build_perseus_geometry,
    spacings_for_count,
)
from apollo14.projector import FovGrid, PlayNitrideLed
from apollo14.units import mm, nm

# ── Which optimizer output, which mirrors ───────────────────────────────────
# REPORT_DIR = None → auto-pick the newest optimize_pupil_perseus run.
# MIRROR_INDICES = None → design every mirror in the stack; or a list for a subset.
OPTIMIZE_RUNS_ROOT = Path("examples/reports/optimize_pupil_perseus")
REPORT_DIR: Path | None = None
MIRROR_INDICES: list[int] | None = None

# ── Coating design knobs (fed to Atlas) ─────────────────────────────────────
REFERENCE_WAVELENGTH_NM = 550.0     # glass index / QWOT seed reference
POLARIZATION = "s"                  # 100% s input; incident p power is zero
NUM_WAVELENGTHS = 120               # target sample points across the band
NUM_ANGLES = 5                      # AOI samples across the swept range

# Alternating high/low-index films (DBR-style contrast) in AGC M-074 glass.
NUM_SEED_FILMS = 11                 # 11 films hold the display fit AND the suppression
# NUM_SEED_FILMS = 15
THICKNESS_BOUNDS_NM = (20.0, 200.0)
TIO2_N_BOUNDS = (2.0, 2.525)        # PLD_TiO2 tunable index range
AL2O3_N_BOUNDS = (1.47, 1.65)       # PLD_Al2O3 tunable index range
SEED_N_HIGH = 2.50                  # QWOT seed indices for the PLD tunable stack
SEED_N_LOW = 1.47
# Fixed-dispersion films used by the active seed (see design_mirror).
HIGH_INDEX_MATERIAL = Materials.balzers.Nb2O5
# LOW_INDEX_MATERIAL = Materials.balzers.Al2O3
LOW_INDEX_MATERIAL = Materials.moveon.SiO2

# The targets are weak (~2%) partial mirrors, so the seed must start in a
# low-reflectance basin. A QWOT H/L stack is a high reflector (seed Rs up to
# ~98%) and L-BFGS gets trapped near it (merit ~1e-2). Instead, seed the high-
# index films thin (at the lower bound) and the low-index spacers at QWOT
# computed from the low-index material's real index.
SEED_THICKNESS_HIGH_NM = 20.0
# Manufacturing thickness (height) tolerance is a hard ±1% of each film's
# nominal thickness. Atlas expresses tolerances as Gaussian 1σ values, so its
# robust optimizer uses one third of that layer-specific hard limit.
THICKNESS_TOLERANCE_PLUS_MINUS_FRACTION = 0.01
ATLAS_TOLERANCE_SIGMA_DIVISOR = 3.0
N_TOLERANCE = 0.005
ROBUST_OPTIMIZATION_SAMPLES = 64
TOLERANCE_VALIDATION_SAMPLES = 1024
TOLERANCE_VALIDATION_BATCH_SIZE = 64
TOLERANCE_MAX_RS_ERROR = 0.01
TOLERANCE_RANDOM_SEED = 42

# Optimizer budget PER MIRROR. Modest defaults so a full stack runs in minutes;
# bump for production recipes (BH max_iterations→100+, local_maxiter→500).
BH_MAX_ITERATIONS = 60
BH_LOCAL_MAXITER = 400
BH_STEPSIZE = 0.2
BH_TEMPERATURE = 0.01
POLISH_MAX_ITERATIONS = 500

# ── Angular suppression (ghosts / see-through) ──────────────────────────────
# Each mirror is a MultiTarget design:
#   display Rs  — match the optimizer's R(λ) for s light at the display AOI band;
#   display Rp  — keep p low there (ambient in the FOV is unpolarized);
#   suppression — unpolarized R as low as possible at SUPPRESSION_REGIONS.
# The regions come from the sun-ray ghost trace (visualize_perseus_sun.py):
# ghosts that TIR off a chassis face reach the eye via a last mirror reflection
# at ~25–36° after a first one at ~66–73°. The angles between them and the
# display band are unconstrained guard bands.
DISPLAY_WAVELENGTHS = "led_bands"   # "led_bands" (emission bands only) | "full" (420–680)
EMISSION_THRESHOLD = 0.10           # led_bands: radiance ≥ this fraction of peak…
BAND_MARGIN_NM = 10.0               # …widened by this margin…
DISPLAY_STEP_NM = 3                 # …sampled every this many nm
DISPLAY_RP_WEIGHT = 0.3
SUPPRESSION_REGIONS = [
    dict(name="suppress 10-36°", angles_deg=(10.0, 36.0), num_angles=14),
    dict(name="suppress 60-75°", angles_deg=(60.0, 75.0), num_angles=16),
]
SUPPRESSION_NUM_WAVELENGTHS = 40
SUPPRESSION_AGGREGATE = "soft_max"  # a ghost is as visible as its worst angle
# An unsuppressed region's loss (~R² ≈ 0.25) dwarfs the display loss (~5e-5),
# so the weight is small; 1e-2 was chosen from the 2026-11-10 log sweep (best
# 10–36° suppression and lowest ghost product, display Rs still ≈ target).
SUPPRESSION_WEIGHT = 1e-2
# Continuation: fit the display alone first, then raise the suppression weight
# through these steps (warm-started, RAMP_HOPS hops each) up to the final one.
SUPPRESSION_WEIGHT_RAMP = (1e-4, 3e-4, 1e-3, 3e-3)
RAMP_HOPS = 8
# Robust basin hopping with the multi-target merit is very slow (hours per
# mirror); the global search runs nominal and the L-BFGS polish runs robust.
ROBUST_BASIN_HOPPING = False
# Print every basin-hopping hop and every suppression stage while a mirror is
# optimized. Off: each mirror prints only its best score once it is done.
VERBOSE_OPTIMIZATION = False

# Per-mirror angular report: R(λ, θ) for s, p and unpolarized on this grid.
ANGULAR_WAVELENGTHS_NM = np.linspace(420.0, 680.0, 131)
ANGULAR_ANGLES_DEG = np.arange(0.0, 85.5, 1.0)
SAVE_ANGULAR_MAPS = True            # store the maps in coating_design.json
# Validated categorical slots (light surface): Rs, Rp, unpolarized.
SERIES_COLORS = {"s": "#2a78d6", "p": "#eb6834", "unpolarized": "#1baf7a"}
SERIES_NAMES = {"s": "Rs", "p": "Rp", "unpolarized": "R unpolarized"}

SAVE_PER_MIRROR_PLOTS = True
OUTPUT_ROOT = "examples/reports"
# GLASS = Materials.moveon.MR10
GLASS = Materials.schott.BASF2

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
    if not elements:
        raise ValueError(f"No PartialMirror elements in {report_dir}")
    elements.sort(key=lambda e: int(e["name"].split("_")[1]))

    mirrors = []
    for mirror_index, element in enumerate(elements):
        element_index = int(element["name"].split("_")[1])
        if element_index != mirror_index:
            raise ValueError(
                f"Mirror indices must be contiguous from zero; found {element['name']}")
        wavelengths_nm = np.asarray(element["wavelengths"]) / float(nm)
        reflectance = np.asarray(element["reflectance"])
        if wavelengths_nm.shape != reflectance.shape:
            raise ValueError(
                f"{element['name']} wavelength/reflectance shapes differ: "
                f"{wavelengths_nm.shape} vs {reflectance.shape}")
        if (not np.all(np.isfinite(wavelengths_nm))
                or not np.all(np.isfinite(reflectance))):
            raise ValueError(f"{element['name']} contains non-finite coating data")
        if np.any((reflectance < 0.0) | (reflectance > 1.0)):
            raise ValueError(f"{element['name']} reflectance lies outside [0, 1]")
        order = np.argsort(wavelengths_nm)
        wavelengths_nm = wavelengths_nm[order]
        if np.any(np.diff(wavelengths_nm) <= 0.0):
            raise ValueError(f"{element['name']} wavelengths are not unique")
        mirrors.append((mirror_index, wavelengths_nm, reflectance[order]))

    reference_wavelengths = mirrors[0][1]
    if any(wavelengths.shape != reference_wavelengths.shape
           or not np.allclose(wavelengths, reference_wavelengths)
           for _, wavelengths, _ in mirrors[1:]):
        raise ValueError("All mirrors must use the same wavelength grid")

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
                                   refractive_index(air, REFERENCE_WAVELENGTH_NM * nm),
                                   glass_index)
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

def projector_spectrum():
    """(wavelengths_nm, radiance normalized to its peak) of the Perseus panel."""
    projector = PlayNitrideLed.create_broadband(
        position=PERSEUS_LIGHT_POSITION, direction=PERSEUS_PROJECTOR_DIRECTION,
        beam_width=PERSEUS_BEAM_WIDTH, beam_height=PERSEUS_BEAM_HEIGHT,
        nx=1, ny=1)
    spec_wavelengths, spec_radiance = projector.spectrum
    spec_radiance = np.asarray(spec_radiance)
    return np.asarray(spec_wavelengths) / float(nm), spec_radiance / spec_radiance.max()


def projector_weights(wavelength_grid_nm) -> list[float]:
    """Weight the fit toward wavelengths the panel actually emits.

    ``0.2 + 0.8·(radiance / peak)`` — every wavelength keeps a ≥0.2 floor so the
    coating stays sane out-of-peak, while the RGB emission bands dominate.
    """
    spec_wavelengths_nm, spec_radiance = projector_spectrum()
    radiance_on_grid = np.interp(wavelength_grid_nm, spec_wavelengths_nm,
                                 spec_radiance)
    return (0.2 + 0.8 * radiance_on_grid / radiance_on_grid.max()).tolist()


def emission_bands(band_nm):
    """Projector emission bands (+ margin) as (start, end) nm, and samples in them."""
    spec_wavelengths_nm, spec_radiance = projector_spectrum()
    grid = np.arange(band_nm[0], band_nm[1] + 1e-9, 1.0)
    emitting = np.interp(grid, spec_wavelengths_nm, spec_radiance) >= EMISSION_THRESHOLD
    widened = np.zeros_like(emitting)
    for center in grid[emitting]:
        widened |= np.abs(grid - center) <= BAND_MARGIN_NM
    bands, start, previous_wavelength = [], None, None
    for wavelength, inside in zip(grid, widened, strict=True):
        if inside and start is None:
            start = wavelength
        if not inside and start is not None:
            bands.append((float(start), float(previous_wavelength)))
            start = None
        previous_wavelength = wavelength
    if start is not None:
        bands.append((float(start), float(grid[-1])))
    return bands, grid[widened][::DISPLAY_STEP_NM]


def band_color_name(band_nm) -> str:
    center = 0.5 * (band_nm[0] + band_nm[1])
    return "blue" if center < 500 else "green" if center < 590 else "red"


def display_sampling(band_nm):
    """Atlas sampling kwargs for the display-Rs region, per DISPLAY_WAVELENGTHS."""
    if DISPLAY_WAVELENGTHS == "led_bands":
        _, points = emission_bands(band_nm)
        return dict(wavelength_points=points.tolist(),
                    weights=projector_weights(points))
    if DISPLAY_WAVELENGTHS == "full":
        return dict(wavelengths=band_nm, num_wavelengths=NUM_WAVELENGTHS,
                    weights=projector_weights(np.linspace(*band_nm, NUM_WAVELENGTHS)))
    raise ValueError(f"DISPLAY_WAVELENGTHS must be 'led_bands' or 'full', "
                     f"got {DISPLAY_WAVELENGTHS!r}")


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


def _atlas_thickness_sigma_nm(thickness_nm: float) -> float:
    """Atlas Gaussian 1σ corresponding to the hard relative thickness limit."""
    return (float(thickness_nm) * THICKNESS_TOLERANCE_PLUS_MINUS_FRACTION
            / ATLAS_TOLERANCE_SIGMA_DIVISOR)


def _seed_layers(glass, seed_thickness_high, seed_thickness_low) -> list:
    """A fresh alternating PLD_TiO2 / PLD_Al2O3 seed stack in ``glass``."""
    layers = [Layer(material=glass)]                       # medium = glass
    for film_idx in range(NUM_SEED_FILMS):
        is_high = film_idx % 2 == 0
        layers.append(Layer(
            material=Materials.pld.TiO2 if is_high else Materials.pld.Al2O3,
            thickness=seed_thickness_high if is_high else seed_thickness_low,
            refractive_index=SEED_N_HIGH if is_high else SEED_N_LOW,
            vary_thickness=True, vary_n=True,
            thickness_bounds=THICKNESS_BOUNDS_NM,
            thickness_tolerance=_atlas_thickness_sigma_nm(
                seed_thickness_high if is_high else seed_thickness_low),
            n_tolerance=N_TOLERANCE,
            n_bounds=TIO2_N_BOUNDS if is_high else AL2O3_N_BOUNDS,
        ))
    layers.append(Layer(material=glass))                   # substrate = glass
    return layers

def _seed_moveon_layers(moveon, seed_thickness_high, seed_thickness_low, glass=GLASS) -> list:
    """moveon materials only."""
    layers = [Layer(material=glass)]
    # stack_materials = [moveon.SiO2, moveon.Ti3O5, moveon.SiO2, moveon.TiO2, moveon.Al2O3, moveon.TiO2, moveon.SiO2, moveon.Ti3O5, moveon.SiO2]
    # stack_materials = [moveon.SiO2, moveon.TiO2, moveon.SiO2, moveon.TiO2, moveon.SiO2, moveon.TiO2, moveon.SiO2, moveon.TiO2, moveon.SiO2, moveon.TiO2, moveon.SiO2, moveon.TiO2, moveon.SiO2]
    # stack_materials = [moveon.TiO2, moveon.SiO2, moveon.TiO2, moveon.SiO2, moveon.TiO2]
    # for i, material in enumerate(stack_materials):
    for i in range(NUM_SEED_FILMS):
        is_tio2 = i % 2 == 1
        layers.append(Layer(
            # material=material,
            material=moveon.TiO2 if is_tio2 else moveon.SiO2,
            thickness=seed_thickness_high if is_tio2 else seed_thickness_low,
            vary_thickness=True,
            thickness_bounds=THICKNESS_BOUNDS_NM,
            thickness_tolerance=_atlas_thickness_sigma_nm(
                seed_thickness_high if is_tio2 else seed_thickness_low),
        ))
    layers.append(Layer(material=glass))

    return layers

def _seed_layers_2_materials(material_1, material_2, thickness_1, thickness_2, glass=GLASS) -> list:
    layers = [Layer(material=glass)]
    for i in range(NUM_SEED_FILMS):
        is_1 = i % 2 == 0
        thickness = thickness_1 if is_1 else thickness_2
        layers.append(Layer(
            material=material_1 if is_1 else material_2,
            thickness=thickness,
            thickness_bounds=THICKNESS_BOUNDS_NM,
            thickness_tolerance=_atlas_thickness_sigma_nm(thickness),
            vary_thickness=True,
        ))
    layers.append(Layer(material=glass))

    return layers

# ── Multi-target spec for one mirror ─────────────────────────────────────────

def build_mirror_target(target_wavelengths_nm, target_reflectance, aoi_range,
                        band_nm, sampling, suppression_weight) -> MultiTarget:
    """Display Rs (match R(λ)) + display Rp (≤ 0) + suppression regions (≤ 0)."""
    def target_curve(wavelength_nm):
        return float(np.interp(wavelength_nm, target_wavelengths_nm, target_reflectance))

    regions = [
        Target(angles=aoi_range, num_angles=NUM_ANGLES, polarization=POLARIZATION,
               reflectance=target_curve, name="display Rs", **sampling),
        Target(wavelengths=band_nm, num_wavelengths=SUPPRESSION_NUM_WAVELENGTHS,
               angles=aoi_range, num_angles=3, polarization="p", reflectance=0.0,
               goal="at_most", weight=DISPLAY_RP_WEIGHT, name="display Rp"),
    ]
    for region in SUPPRESSION_REGIONS:
        regions.append(Target(
            wavelengths=band_nm, num_wavelengths=SUPPRESSION_NUM_WAVELENGTHS,
            angles=region["angles_deg"], num_angles=region["num_angles"],
            polarization="both", reflectance=0.0, goal="at_most",
            aggregate=SUPPRESSION_AGGREGATE, weight=suppression_weight,
            name=region["name"]))
    return MultiTarget(regions=regions)


def _rebase_tolerances(layers):
    """Rebase each film's ±1% sigma on its current thickness (in place)."""
    for layer in layers[1:-1]:
        layer.thickness_tolerance = _atlas_thickness_sigma_nm(layer.thickness)
    return layers


# ── Design one mirror (display fit → suppression ramp → robust polish) ───────

def design_mirror(mirror_index, target_wavelengths_nm, target_reflectance,
                  aoi_range, band_nm, sampling, glass, seed_thickness_high,
                  seed_thickness_low):
    """Staged Atlas design for one mirror; return its (robust-polished) result.

    1. basin hopping on the display alone (suppression weight 0);
    2. warm-started basin hopping through ``SUPPRESSION_WEIGHT_RAMP`` up to
       ``SUPPRESSION_WEIGHT`` — raising the weight gradually keeps the design in
       the display-fitting basin instead of jumping straight to suppression;
    3. robust L-BFGS polish of the final multi-target merit under the ±1%
       thickness tolerance.
    """
    layers = _seed_layers_2_materials(
        material_1=HIGH_INDEX_MATERIAL, material_2=LOW_INDEX_MATERIAL,
        thickness_1=seed_thickness_high, thickness_2=seed_thickness_low, glass=glass)
    stages = ([(0.0, BH_MAX_ITERATIONS, "display only")]
              + [(weight, RAMP_HOPS, f"ramp {weight:g}") for weight in SUPPRESSION_WEIGHT_RAMP]
              + [(SUPPRESSION_WEIGHT, RAMP_HOPS, f"final {SUPPRESSION_WEIGHT:g}")])
    stage_log = []
    for suppression_weight, hops, label in stages:
        target = build_mirror_target(target_wavelengths_nm, target_reflectance,
                                     aoi_range, band_nm, sampling, suppression_weight)
        designer = OpticalDesigner(layers=layers, target=target)
        hop_state = {"hop": 0, "best": float("inf")}

        def hop_progress(x, merit, accept, hop_state=hop_state, label=label):
            hop_state["hop"] += 1
            merit = float(merit)
            marker = "★" if merit < hop_state["best"] else " "
            hop_state["best"] = min(hop_state["best"], merit)
            if VERBOSE_OPTIMIZATION and label == "display only":
                print(f"    mirror {mirror_index}: {label} hop {hop_state['hop']:>3} "
                      f" merit={merit:.4e} best={hop_state['best']:.4e} {marker}", flush=True)

        result_bh = designer.optimize(
            method="basin_hopping", max_iterations=hops,
            robust=ROBUST_BASIN_HOPPING, num_samples=ROBUST_OPTIMIZATION_SAMPLES,
            stepsize=BH_STEPSIZE, temperature=BH_TEMPERATURE,
            local_maxiter=BH_LOCAL_MAXITER, callback=hop_progress)
        regions = {region.name: region for region in result_bh.regions}
        stage_log.append(dict(
            stage=label, suppression_weight=suppression_weight, hops=hops,
            merit=float(result_bh.merit),
            worst_violation={name: float(region.worst_violation)
                             for name, region in regions.items()}))
        if VERBOSE_OPTIMIZATION:
            print(f"    mirror {mirror_index}: {label:<12} merit={result_bh.merit:.4e}  "
                  + "  ".join(f"{name} worst {100 * region.worst_violation:.2f}"
                              + ("pp" if region.goal == "match" else "%")
                              for name, region in regions.items()), flush=True)
        layers = _rebase_tolerances(result_bh.layers)

    final_target = designer.target
    polished_designer = OpticalDesigner(layers=layers, target=final_target)
    result_polished = polished_designer.optimize(
        method="lbfgs", max_iterations=POLISH_MAX_ITERATIONS,
        robust=True, num_samples=ROBUST_OPTIMIZATION_SAMPLES)
    if ROBUST_BASIN_HOPPING and result_bh.merit < result_polished.merit:
        # Both merits are robust, so they are comparable.
        result = result_bh
    else:
        # A nominal basin merit is not comparable with the robust polish:
        # the robust design is the one meant for manufacture.
        result = result_polished
    result.selected_stage = "polish" if result is result_polished else "basin_hopping"
    result.basin_merit = float(result_bh.merit)
    result.polish_merit = float(result_polished.merit)
    result.stages = stage_log
    # Ensure the exported recipe records ±1% relative to its final thickness.
    _rebase_tolerances(result.layers)

    stack = Stack(layers=result.layers)
    reflectance, transmittance = compute_rt_polarized(
        stack,
        jnp.asarray(result.wavelengths_nm) * 1e-9,
        jnp.deg2rad(jnp.asarray(result.angles_deg)),
        stack.get_thickness_array(),
        stack.get_tunable_n_array(),
        _precompute_fixed_nk_data(stack),
    )
    result.reflectance = reflectance[..., 0]
    result.p_reflectance = reflectance[..., 1]
    result.s_transmittance = transmittance[..., 0]
    result.p_transmittance = transmittance[..., 1]
    return result


# ── Per-mirror angular report ────────────────────────────────────────────────

def _angle_mask(low_deg, high_deg):
    return (low_deg <= ANGULAR_ANGLES_DEG) & (high_deg >= ANGULAR_ANGLES_DEG)


def angular_report(result, aoi_range, bands) -> dict:
    """R(λ, θ) over 0–85° for s, p and unpolarized, plus region metrics.

    ``display`` — projector-weighted Rs over the FOV and per LED color, Rp and
    unpolarized R at the display angles; ``regions`` — worst / mean
    reflectance in each suppression region and where the worst sits;
    ``ghost_product`` — product of the regions' worst (and mean) unpolarized R,
    which is how a two-reflection TIR ghost scales; ``unconstrained_low`` —
    reflectance below the first suppression region, which no target covers.
    """
    stack = Stack(layers=result.layers)
    reflectance, _ = compute_rt_polarized(
        stack, jnp.asarray(ANGULAR_WAVELENGTHS_NM) * 1e-9,
        jnp.deg2rad(jnp.asarray(ANGULAR_ANGLES_DEG)),
        stack.get_thickness_array(), stack.get_tunable_n_array(),
        _precompute_fixed_nk_data(stack))
    reflectance = np.asarray(reflectance)
    maps = {"s": reflectance[..., 0], "p": reflectance[..., 1]}
    maps["unpolarized"] = 0.5 * (maps["s"] + maps["p"])

    spec_wavelengths_nm, spec_radiance = projector_spectrum()
    spectrum_weights = np.interp(ANGULAR_WAVELENGTHS_NM, spec_wavelengths_nm, spec_radiance)
    display = _angle_mask(*aoi_range)
    weighted_rs = (spectrum_weights @ maps["s"][:, display]) / spectrum_weights.sum()
    per_color = {}
    for band in bands:
        in_band = (band[0] <= ANGULAR_WAVELENGTHS_NM) & (band[1] >= ANGULAR_WAVELENGTHS_NM)
        band_weights = spectrum_weights[in_band]
        band_rs = maps["s"][in_band][:, display]
        per_color[band_color_name(band)] = dict(
            band_nm=list(band),
            rs_weighted_mean=float((band_weights @ band_rs).mean() / band_weights.sum()),
            rs_min=float(band_rs.min()), rs_max=float(band_rs.max()),
            rp_mean=float(maps["p"][in_band][:, display].mean()))

    regions = {}
    for region in SUPPRESSION_REGIONS:
        mask = _angle_mask(*region["angles_deg"])
        unpolarized = maps["unpolarized"][:, mask]
        worst_wl, worst_angle = np.unravel_index(unpolarized.argmax(), unpolarized.shape)
        regions[region["name"]] = dict(
            angles_deg=list(region["angles_deg"]),
            worst_unpolarized=float(unpolarized.max()),
            mean_unpolarized=float(unpolarized.mean()),
            worst_at_nm=float(ANGULAR_WAVELENGTHS_NM[worst_wl]),
            worst_at_deg=float(ANGULAR_ANGLES_DEG[mask][worst_angle]),
            worst_s=float(maps["s"][:, mask].max()),
            worst_p=float(maps["p"][:, mask].max()),
            fraction_above_3pct=float(np.mean(unpolarized > 0.03)))
    low_edge = min(region["angles_deg"][0] for region in SUPPRESSION_REGIONS)
    below = low_edge > ANGULAR_ANGLES_DEG
    metrics = dict(
        display=dict(
            rs_weighted_range=[float(weighted_rs.min()), float(weighted_rs.max())],
            rs_weighted_mean=float(weighted_rs.mean()),
            per_color=per_color,
            rp_max=float(maps["p"][:, display].max()),
            unpolarized_mean=float(maps["unpolarized"][:, display].mean())),
        regions=regions,
        ghost_product=dict(
            worst=float(np.prod([value["worst_unpolarized"] for value in regions.values()])),
            mean=float(np.prod([value["mean_unpolarized"] for value in regions.values()]))),
        unconstrained_low=dict(
            angles_deg=[0.0, float(low_edge)],
            max_unpolarized=float(maps["unpolarized"][:, below].max()) if below.any() else None,
            mean_unpolarized=float(maps["unpolarized"][:, below].mean()) if below.any() else None),
    )
    return dict(metrics=metrics, maps=maps)


def _print_angular_summary(metrics) -> None:
    display = metrics["display"]
    print(f"    angular: display Rs {100 * display['rs_weighted_range'][0]:.2f}–"
          f"{100 * display['rs_weighted_range'][1]:.2f}% ("
          + ", ".join(f"{color} {100 * values['rs_weighted_mean']:.2f}%"
                      for color, values in display["per_color"].items())
          + f"), Rp max {100 * display['rp_max']:.2f}%, "
            f"unpolarized {100 * display['unpolarized_mean']:.2f}%")
    for name, values in metrics["regions"].items():
        print(f"             {name}: worst {100 * values['worst_unpolarized']:.2f}% "
              f"@ {values['worst_at_nm']:.0f} nm/{values['worst_at_deg']:.0f}°, "
              f"mean {100 * values['mean_unpolarized']:.2f}% "
              f"(Rs {100 * values['worst_s']:.2f}%, Rp {100 * values['worst_p']:.2f}%)")
    low = metrics["unconstrained_low"]
    if low["max_unpolarized"] is not None:
        print(f"             unconstrained {low['angles_deg'][0]:.0f}–{low['angles_deg'][1]:.0f}°: "
              f"max {100 * low['max_unpolarized']:.2f}%, mean {100 * low['mean_unpolarized']:.2f}%;"
              f" ghost product worst {metrics['ghost_product']['worst']:.4f}")


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
        "thickness_tolerance_plus_minus_percent": (
            100.0 * THICKNESS_TOLERANCE_PLUS_MINUS_FRACTION),
        "atlas_thickness_sigma_nm": float(layer.thickness_tolerance),
        "refractive_index": (None if layer.refractive_index is None
                             else float(layer.refractive_index)),
        "n": float(nk.real),
        "k": float(nk.imag),
        "is_tunable": bool(layer.material.is_tunable),
    }


def save_mirror_plot_html(result, angular, aoi_range, path: Path,
                          config_text: str) -> None:
    """Interactive per-mirror report, as a Plotly HTML file.

    1. **Display fit** — Rs/Rp vs wavelength at each display angle against the
       target (replaces Atlas's ``plot_per_angle`` PNG; axes autoscale, so the
       ~2–16% curves are not buried on a 0–1 axis);
    2. **Angular map** — unpolarized R(θ, λ) over 0–85°, with the display band
       and the suppression regions marked;
    3. **Angular profile** — Rs, Rp and unpolarized R vs AOI, weighted by the
       projector spectrum.
    """
    wavelengths_nm = np.asarray(result.wavelengths_nm)
    target = np.asarray(result.target_reflectance)
    achieved_s = np.asarray(result.reflectance)
    achieved_p = np.asarray(result.p_reflectance)
    if achieved_s.ndim == 1:
        achieved_s = achieved_s[:, np.newaxis]
        achieved_p = achieved_p[:, np.newaxis]
    angles_deg = np.asarray(result.angles_deg)
    if len(angles_deg) > 1:
        angle_span = (f"{float(angles_deg[0]):.1f}°–{float(angles_deg[-1]):.1f}° "
                      f"({len(angles_deg)} samples)")
    else:
        angle_span = f"{float(angles_deg[0]):.1f}°"

    figure = make_subplots(
        rows=3, cols=1, vertical_spacing=0.08, row_heights=[0.36, 0.36, 0.28],
        subplot_titles=(
            f"Display fit — S/P reflectance vs target at {angle_span}",
            "Angular map — unpolarized R(θ, λ)",
            "Angular profile — reflectance vs AOI (projector-weighted)"))

    # 1. Display fit: one trace per angle of incidence, coloured across the range.
    colors = sample_colorscale(
        "Viridis", np.linspace(0.1, 0.9, achieved_s.shape[1]).tolist())
    for angle_idx, (angle, color) in enumerate(zip(angles_deg, colors, strict=True)):
        figure.add_trace(go.Scatter(
            x=wavelengths_nm, y=achieved_s[:, angle_idx], mode="lines+markers",
            marker=dict(size=3), name=f"S · {float(angle):.1f}°", legendgroup="display",
            line=dict(color=color, width=1.5),
            hovertemplate="%{x:.1f} nm<br>R=%{y:.4f}<extra>%{fullData.name}</extra>"),
            row=1, col=1)
        figure.add_trace(go.Scatter(
            x=wavelengths_nm, y=achieved_p[:, angle_idx], mode="lines",
            name=f"P · {float(angle):.1f}°", legendgroup="display",
            line=dict(color=color, width=1.5, dash="dot"),
            hovertemplate="%{x:.1f} nm<br>R=%{y:.4f}<extra>%{fullData.name}</extra>"),
            row=1, col=1)
    # Target last so it draws on top of the achieved family.
    figure.add_trace(go.Scatter(
        x=wavelengths_nm, y=target, mode="lines", name="S target", legendgroup="display",
        line=dict(color="black", width=2.5, dash="dash"),
        hovertemplate="%{x:.1f} nm<br>target R=%{y:.4f}<extra>Target</extra>"),
        row=1, col=1)

    # 2. Angular map — sequential single hue (magnitude).
    maps = angular["maps"]
    figure.add_trace(go.Heatmap(
        x=ANGULAR_ANGLES_DEG, y=ANGULAR_WAVELENGTHS_NM, z=100 * maps["unpolarized"],
        colorscale=[[0, "#fcfcfb"], [0.25, "#9ec5f4"], [0.5, "#3987e5"],
                    [0.75, "#1c5cab"], [1, "#0d366b"]],
        colorbar=dict(title="R %", len=0.33, y=0.5),
        hovertemplate="AOI %{x:.0f}°<br>%{y:.0f} nm<br>R %{z:.2f}%<extra></extra>"),
        row=2, col=1)

    # 3. Angular profile, projector-weighted over wavelength.
    spec_wavelengths_nm, spec_radiance = projector_spectrum()
    spectrum_weights = np.interp(ANGULAR_WAVELENGTHS_NM, spec_wavelengths_nm, spec_radiance)
    for polarization in ("s", "p", "unpolarized"):
        curve = 100 * (spectrum_weights @ maps[polarization]) / spectrum_weights.sum()
        figure.add_trace(go.Scatter(
            x=ANGULAR_ANGLES_DEG, y=curve, mode="lines", legendgroup="angular",
            line=dict(color=SERIES_COLORS[polarization], width=2),
            name=SERIES_NAMES[polarization],
            hovertemplate="AOI %{x:.0f}°<br>%{y:.2f}%<extra>"
                          + SERIES_NAMES[polarization] + "</extra>"),
            row=3, col=1)
        # Direct label at the line end (identity is never color-alone).
        figure.add_annotation(x=float(ANGULAR_ANGLES_DEG[-1]), y=float(curve[-1]),
                              text=SERIES_NAMES[polarization], showarrow=False,
                              xanchor="left", font=dict(color="#52514e", size=11),
                              row=3, col=1)
    for row in (2, 3):
        figure.add_vrect(x0=aoi_range[0], x1=aoi_range[1], row=row, col=1,
                         fillcolor="rgba(42,120,214,0.12)", line_width=0,
                         annotation_text="display" if row == 3 else None,
                         annotation_position="top left")
        for region in SUPPRESSION_REGIONS:
            figure.add_vrect(x0=region["angles_deg"][0], x1=region["angles_deg"][1],
                             row=row, col=1, fillcolor="rgba(82,81,78,0.08)",
                             line=dict(color="#52514e", width=1, dash="dot"),
                             annotation_text="suppress" if row == 3 else None,
                             annotation_position="top left")

    metrics = angular["metrics"]
    angular_text = "<br>".join(
        [f"display Rs {100 * metrics['display']['rs_weighted_range'][0]:.2f}–"
         f"{100 * metrics['display']['rs_weighted_range'][1]:.2f}%, "
         f"Rp max {100 * metrics['display']['rp_max']:.2f}%"]
        + [f"{name}: worst {100 * values['worst_unpolarized']:.2f}%, "
           f"mean {100 * values['mean_unpolarized']:.2f}%"
           for name, values in metrics["regions"].items()])
    figure.add_annotation(
        text=config_text.replace("\n", "<br>") + "<br>" + angular_text,
        xref="paper", yref="paper", x=0.01, y=1.0, yanchor="bottom",
        showarrow=False, align="left", font=dict(family="monospace", size=11),
        bgcolor="rgba(255,255,255,0.85)", bordercolor="#bbbbbb", borderwidth=1)

    figure.update_xaxes(title_text="Wavelength (nm)", row=1, col=1)
    # Free y-axis: the point of the HTML version is that the reader zooms
    # instead of the author guessing a range.
    figure.update_yaxes(title_text="Reflectance", tickformat=".3f", row=1, col=1)
    figure.update_xaxes(title_text="AOI in glass (°)", range=[0, 85], row=2, col=1)
    figure.update_yaxes(title_text="Wavelength (nm)", row=2, col=1)
    figure.update_xaxes(title_text="AOI in glass (°)", range=[0, 93], row=3, col=1)
    figure.update_yaxes(title_text="Reflectance (%)", rangemode="tozero", row=3, col=1)
    figure.update_layout(
        title=f"Perseus mirror coating — merit={result.merit:.3e}",
        height=1350, margin=dict(t=170), hovermode="closest", template="plotly_white",
        paper_bgcolor="#fcfcfb", plot_bgcolor="#fcfcfb", font=dict(color="#0b0b0b"))
    figure.write_html(str(path), include_plotlyjs="cdn")


def _validation_metrics(result) -> dict:
    """Summarize polarization, angular error, and power conservation."""
    target = np.asarray(result.target_reflectance)[:, np.newaxis]
    angles = np.asarray(result.angles_deg)
    rs = np.asarray(result.reflectance)
    rp = np.asarray(result.p_reflectance)
    ts = np.asarray(result.s_transmittance)
    tp = np.asarray(result.p_transmittance)
    if rs.ndim == 1:
        rs = rs[:, np.newaxis]
        rp = rp[:, np.newaxis]
        ts = ts[:, np.newaxis]
        tp = tp[:, np.newaxis]

    s_error = rs - target
    absorption_s = 1.0 - rs - ts
    absorption_p = 1.0 - rp - tp
    rms_by_angle = np.sqrt(np.mean(np.square(s_error), axis=0))
    worst_angle_idx = int(np.argmax(rms_by_angle))

    per_angle = []
    for angle_idx, angle_deg in enumerate(angles):
        angle_error = s_error[:, angle_idx]
        per_angle.append({
            "angle_deg": float(angle_deg),
            "s_rmse": float(np.sqrt(np.mean(np.square(angle_error)))),
            "s_max_abs_error": float(np.max(np.abs(angle_error))),
            "s_mean_reflectance": float(np.mean(rs[:, angle_idx])),
            "p_mean_reflectance": float(np.mean(rp[:, angle_idx])),
            "p_max_reflectance": float(np.max(rp[:, angle_idx])),
            "s_mean_transmittance": float(np.mean(ts[:, angle_idx])),
            "p_mean_transmittance": float(np.mean(tp[:, angle_idx])),
            "s_mean_absorption": float(np.mean(absorption_s[:, angle_idx])),
            "p_mean_absorption": float(np.mean(absorption_p[:, angle_idx])),
        })

    return {
        "s_target_rmse": float(np.sqrt(np.mean(np.square(s_error)))),
        "s_target_mean_abs_error": float(np.mean(np.abs(s_error))),
        "s_target_max_abs_error": float(np.max(np.abs(s_error))),
        "worst_angle_deg": float(angles[worst_angle_idx]),
        "worst_angle_s_rmse": float(rms_by_angle[worst_angle_idx]),
        "p_mean_reflectance": float(np.mean(rp)),
        "p_max_reflectance": float(np.max(rp)),
        "s_mean_transmittance": float(np.mean(ts)),
        "p_mean_transmittance": float(np.mean(tp)),
        "s_absorption_mean": float(np.mean(absorption_s)),
        "s_absorption_max": float(np.max(absorption_s)),
        "p_absorption_mean": float(np.mean(absorption_p)),
        "p_absorption_max": float(np.max(absorption_p)),
        "s_reflectance_plus_transmittance_min": float(np.min(rs + ts)),
        "s_reflectance_plus_transmittance_max": float(np.max(rs + ts)),
        "p_reflectance_plus_transmittance_min": float(np.min(rp + tp)),
        "p_reflectance_plus_transmittance_max": float(np.max(rp + tp)),
        "per_angle": per_angle,
    }


def validate_manufacturing_tolerance(result, *, seed: int, band_nm) -> dict:
    """Evaluate bounded thickness and Gaussian index manufacturing errors.

    Every sample is scored on the display (max |Rs − target|, Rp) and on each
    suppression region (worst unpolarized R), so the yield reflects the
    display spec and the percentiles show how robust the suppression is.
    """
    if TOLERANCE_VALIDATION_SAMPLES <= 0 or (
        TOLERANCE_VALIDATION_SAMPLES & (TOLERANCE_VALIDATION_SAMPLES - 1)
    ):
        raise ValueError("TOLERANCE_VALIDATION_SAMPLES must be a positive power of two")
    if TOLERANCE_VALIDATION_BATCH_SIZE <= 0:
        raise ValueError("TOLERANCE_VALIDATION_BATCH_SIZE must be positive")

    stack = Stack(layers=result.layers)
    base_thicknesses = np.asarray(stack.get_thickness_array())
    base_indices = np.asarray(stack.get_tunable_n_array())
    num_thicknesses = base_thicknesses.size
    num_indices = base_indices.size
    dimension = num_thicknesses + num_indices

    sampler = qmc.Sobol(d=dimension, scramble=True, seed=seed)
    unit_samples = sampler.random_base2(
        m=int(np.log2(TOLERANCE_VALIDATION_SAMPLES)))
    thickness_limits_nm = (
        base_thicknesses * THICKNESS_TOLERANCE_PLUS_MINUS_FRACTION)
    thickness_error = (
        2.0 * unit_samples[:, :num_thicknesses] - 1.0
    ) * thickness_limits_nm[None, :]
    thickness_samples = base_thicknesses[None, :] + thickness_error

    if num_indices:
        # Convert the remaining Sobol coordinates to a zero-mean Gaussian for
        # Atlas's refractive-index 1σ tolerance. Clipping avoids norm.ppf(0/1).
        index_unit = np.clip(
            unit_samples[:, num_thicknesses:], 1e-12, 1.0 - 1e-12)
        index_error = norm.ppf(index_unit) * N_TOLERANCE
        index_samples = base_indices[None, :] + index_error
    else:
        index_samples = np.empty((TOLERANCE_VALIDATION_SAMPLES, 0))

    wavelengths = jnp.asarray(result.wavelengths_nm) * 1e-9
    angles = jnp.deg2rad(jnp.asarray(result.angles_deg))
    fixed_nk_data = _precompute_fixed_nk_data(stack)
    target = np.asarray(result.target_reflectance)[:, None]

    suppression_grids = [
        (jnp.asarray(np.linspace(*band_nm, SUPPRESSION_NUM_WAVELENGTHS)) * 1e-9,
         jnp.deg2rad(jnp.linspace(*region["angles_deg"], region["num_angles"])))
        for region in SUPPRESSION_REGIONS]

    def evaluate_one(thicknesses, indices):
        reflectance, _ = compute_rt_polarized(
            stack, wavelengths, angles, thicknesses, indices, fixed_nk_data)
        suppression_worst = []
        for region_wavelengths, region_angles in suppression_grids:
            region_reflectance, _ = compute_rt_polarized(
                stack, region_wavelengths, region_angles, thicknesses, indices,
                fixed_nk_data)
            suppression_worst.append(jnp.max(
                0.5 * (region_reflectance[..., 0] + region_reflectance[..., 1])))
        return reflectance, jnp.stack(suppression_worst)

    evaluate_batch = jax.jit(jax.vmap(evaluate_one))
    sample_rmse = []
    sample_max_error = []
    sample_p_max = []
    sample_suppression_worst = []
    for start in range(0, TOLERANCE_VALIDATION_SAMPLES,
                       TOLERANCE_VALIDATION_BATCH_SIZE):
        stop = min(start + TOLERANCE_VALIDATION_BATCH_SIZE,
                   TOLERANCE_VALIDATION_SAMPLES)
        reflectance, suppression_worst = evaluate_batch(
            jnp.asarray(thickness_samples[start:stop]),
            jnp.asarray(index_samples[start:stop]))
        reflectance = np.asarray(reflectance)
        sample_suppression_worst.append(np.asarray(suppression_worst))
        s_error = reflectance[..., 0] - target[None, :, :]
        sample_rmse.append(np.sqrt(np.mean(np.square(s_error), axis=(1, 2))))
        sample_max_error.append(np.max(np.abs(s_error), axis=(1, 2)))
        sample_p_max.append(np.max(reflectance[..., 1], axis=(1, 2)))

    sample_rmse = np.concatenate(sample_rmse)
    sample_max_error = np.concatenate(sample_max_error)
    sample_p_max = np.concatenate(sample_p_max)
    sample_suppression_worst = np.concatenate(sample_suppression_worst)
    percentiles = [50, 90, 95, 99, 100]

    return {
        "samples": TOLERANCE_VALIDATION_SAMPLES,
        "sampling": "scrambled Sobol",
        "seed": seed,
        "thickness_error_distribution": "uniform bounded",
        "thickness_plus_minus_fraction": (
            THICKNESS_TOLERANCE_PLUS_MINUS_FRACTION),
        "thickness_plus_minus_percent": (
            100.0 * THICKNESS_TOLERANCE_PLUS_MINUS_FRACTION),
        "per_layer_thickness_plus_minus_nm": thickness_limits_nm.tolist(),
        "index_error_distribution": "Gaussian",
        "index_sigma": N_TOLERANCE,
        "max_rs_error_limit": TOLERANCE_MAX_RS_ERROR,
        "yield_fraction": float(np.mean(
            sample_max_error <= TOLERANCE_MAX_RS_ERROR)),
        "rs_rmse_percentiles": {
            str(p): float(np.percentile(sample_rmse, p)) for p in percentiles
        },
        "rs_max_abs_error_percentiles": {
            str(p): float(np.percentile(sample_max_error, p)) for p in percentiles
        },
        "rp_max_percentiles": {
            str(p): float(np.percentile(sample_p_max, p)) for p in percentiles
        },
        "suppression_worst_unpolarized_percentiles": {
            region["name"]: {
                str(p): float(np.percentile(sample_suppression_worst[:, region_idx], p))
                for p in percentiles}
            for region_idx, region in enumerate(SUPPRESSION_REGIONS)
        },
    }


def _deposition_to_dict(film_layers) -> dict:
    """Explicit layer ordering for the coater.

    Atlas orders films from the incident medium toward the substrate, so
    ``layers[0]`` is the film the design light meets first and ``layers[-1]``
    touches the substrate. Coating grows *on* the substrate, so deposition runs
    in reverse: ``layers[-1]`` is deposited first.
    """
    is_lossless = all(float(layer.index_at(REFERENCE_WAVELENGTH_NM).imag) == 0.0
                      for layer in film_layers)
    deposition_sequence = [
        {"step": step,
         "layer_index": layer_index,
         "material": film_layers[layer_index].material.name,
         "thickness_nm": float(film_layers[layer_index].thickness)}
        for step, layer_index in enumerate(reversed(range(len(film_layers))))
    ]
    return {
        "layer_order": ("layers[0] faces the incident medium (first film the "
                        "design light hits); layers[-1] touches the substrate"),
        "deposition_order": ("deposited on the substrate starting from "
                             "layers[-1] and ending with layers[0]"),
        # Lossless films between identical glass give the same R/T from either
        # side, so which physical face meets the projector does not matter.
        "orientation_sensitive": not is_lossless,
        "deposition_sequence": deposition_sequence,
    }


def _result_to_dict(result) -> dict:
    return {
        "merit": float(result.merit),
        "selected_stage": result.selected_stage,
        "basin_merit": result.basin_merit,
        "polish_merit": result.polish_merit,
        "nominal_merit": (None if result.nominal_merit is None
                          else float(result.nominal_merit)),
        "converged": bool(result.converged),
        "num_films": len(result.film_layers),
        "incident_medium": result.layers[0].material.name,
        "substrate": result.layers[-1].material.name,
        "layers": [_layer_to_dict(layer) for layer in result.film_layers],
        "deposition": _deposition_to_dict(result.film_layers),
        "achieved_reflectance": {
            "wavelengths_nm": np.asarray(result.wavelengths_nm).tolist(),
            "angles_deg": np.asarray(result.angles_deg).tolist(),
            "values": np.asarray(result.reflectance).tolist(),   # Rs legacy key
            "s_values": np.asarray(result.reflectance).tolist(),
            "p_values": np.asarray(result.p_reflectance).tolist(),
        },
        "achieved_transmittance": {
            "s_values": np.asarray(result.s_transmittance).tolist(),
            "p_values": np.asarray(result.p_transmittance).tolist(),
        },
        "validation": _validation_metrics(result),
        "tolerance_validation": result.tolerance_validation,
        "stages": result.stages,
        "regions": [region.to_dict() for region in result.regions],
        "angular": {
            "metrics": result.angular["metrics"],
            **({"maps": {
                "wavelengths_nm": ANGULAR_WAVELENGTHS_NM.tolist(),
                "angles_deg": ANGULAR_ANGLES_DEG.tolist(),
                **{f"R_{polarization}": values.tolist()
                   for polarization, values in result.angular["maps"].items()},
            }} if SAVE_ANGULAR_MAPS else {}),
        },
    }


# ── Run ──────────────────────────────────────────────────────────────────────

def main():
    report_dir = REPORT_DIR or _latest_report_dir()
    (mirrors, num_mirrors, curve_mode, size_wh_mm, mirror_normal,
     source_optimizer) = load_report(report_dir)

    indices = (list(range(num_mirrors)) if MIRROR_INDICES is None
               else MIRROR_INDICES)
    invalid_indices = [index for index in indices
                       if index < 0 or index >= num_mirrors]
    if invalid_indices:
        raise ValueError(
            f"MIRROR_INDICES contains invalid entries {invalid_indices}; "
            f"valid range is 0..{num_mirrors - 1}")

    glass = GLASS
    # Geometry / spectrum shared by the whole stack (mirrors share one normal).
    glass_index = float(refractive_index(glass, REFERENCE_WAVELENGTH_NM * nm))
    aoi_min, aoi_max, aoi_on_axis, fov_ray_angles = incidence_angle_range(
        mirror_normal, num_mirrors, glass_index)
    aoi_design = 0.5 * (aoi_min + aoi_max)
    high_index = float(Layer(material=HIGH_INDEX_MATERIAL, thickness=1.0)
                       .index_at(REFERENCE_WAVELENGTH_NM).real)
    low_index = float(Layer(material=LOW_INDEX_MATERIAL, thickness=1.0)
                      .index_at(REFERENCE_WAVELENGTH_NM).real)
    seed_thickness_high = SEED_THICKNESS_HIGH_NM
    seed_thickness_low = _qwot_thickness_nm(
        REFERENCE_WAVELENGTH_NM, low_index, glass_index, aoi_design)

    band = (float(mirrors[0][1].min()), float(mirrors[0][1].max()))
    bands, _ = emission_bands(band)
    sampling = display_sampling(band)
    num_display_wavelengths = len(sampling.get("wavelength_points", ())) or NUM_WAVELENGTHS

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
          f"{HIGH_INDEX_MATERIAL.name}/{LOW_INDEX_MATERIAL.name}, "
          f"thin H={seed_thickness_high:.0f}nm, "
          f"L=QWOT@{aoi_design:.0f}° {seed_thickness_low:.0f}nm")
    print(f"display λ     : {DISPLAY_WAVELENGTHS} — "
          + (", ".join(f"{start:.0f}–{end:.0f} nm" for start, end in bands)
             if DISPLAY_WAVELENGTHS == "led_bands" else f"{band[0]:.0f}–{band[1]:.0f} nm")
          + f" ({num_display_wavelengths} samples, projector-weighted); Rp ≤ 0 "
            f"(weight {DISPLAY_RP_WEIGHT:g})")
    print("suppression   : " + ", ".join(
        f"{region['angles_deg'][0]:.0f}–{region['angles_deg'][1]:.0f}°"
        for region in SUPPRESSION_REGIONS)
          + f", unpolarized, {band[0]:.0f}–{band[1]:.0f} nm, weight {SUPPRESSION_WEIGHT:g} "
            f"(ramp {', '.join(f'{w:g}' for w in SUPPRESSION_WEIGHT_RAMP)})")
    print(f"budget/mirror : basin_hopping {BH_MAX_ITERATIONS}×{BH_LOCAL_MAXITER} display-only"
          f"{' (robust)' if ROBUST_BASIN_HOPPING else ' (nominal)'}, "
          f"{len(SUPPRESSION_WEIGHT_RAMP) + 1}×{RAMP_HOPS} ramp hops, "
          f"robust lbfgs polish {POLISH_MAX_ITERATIONS}\n")

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
            band, sampling, glass, seed_thickness_high, seed_thickness_low)
        result.angular = angular_report(result, (aoi_min, aoi_max), bands)
        display_only, final_stage = result.stages[0], result.stages[-1]
        print(f"    best score: {result.merit:.4e} ({result.selected_stage}; "
              f"display-only basin best {display_only['merit']:.4e}, "
              f"final-weight basin best {final_stage['merit']:.4e}, "
              f"robust polish {result.polish_merit:.4e})", flush=True)
        validation = _validation_metrics(result)
        print(f"    → merit {result.merit:.4e}, {len(result.film_layers)} films, "
              f"Rs RMSE={validation['s_target_rmse']:.4e}, "
              f"worst={validation['s_target_max_abs_error']:.4e}, "
              f"Rp mean/max={validation['p_mean_reflectance']:.4f}/"
              f"{validation['p_max_reflectance']:.4f}")
        _print_angular_summary(result.angular["metrics"])
        print(f"    tolerance validation: {TOLERANCE_VALIDATION_SAMPLES} samples")
        result.tolerance_validation = validate_manufacturing_tolerance(
            result, seed=TOLERANCE_RANDOM_SEED + mirror_index, band_nm=band)
        print(f"    → tolerance yield="
              f"{result.tolerance_validation['yield_fraction']:.1%} "
              f"at max |Rs-target| ≤ {TOLERANCE_MAX_RS_ERROR:.3f}; suppression worst R "
              f"median / 95th percentile: "
              + ", ".join(f"{name} {100 * values['50']:.2f}% / {100 * values['95']:.2f}%"
                          for name, values in result.tolerance_validation[
                              "suppression_worst_unpolarized_percentiles"].items()))

        if SAVE_PER_MIRROR_PLOTS:
            save_mirror_plot_html(
                result, result.angular, (aoi_min, aoi_max),
                out_dir / f"coating_mirror_{mirror_index}.html",
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
            "display_wavelengths": DISPLAY_WAVELENGTHS,
            "display_wavelength_bands_nm": bands,
            "num_wavelengths": num_display_wavelengths,
            "num_angles": NUM_ANGLES,
            "fit_weights": "0.2 + 0.8·radiance/peak (projector-spectrum weighted)",
            "angular_suppression": {
                "display_rp": {"goal": "at_most 0", "weight": DISPLAY_RP_WEIGHT},
                "regions": [dict(name=region["name"], angles_deg=list(region["angles_deg"]),
                                 num_angles=region["num_angles"], polarization="both",
                                 wavelengths_nm=list(band),
                                 num_wavelengths=SUPPRESSION_NUM_WAVELENGTHS,
                                 goal="at_most 0", aggregate=SUPPRESSION_AGGREGATE)
                            for region in SUPPRESSION_REGIONS],
                "weight": SUPPRESSION_WEIGHT,
                "weight_ramp": list(SUPPRESSION_WEIGHT_RAMP),
                "ramp_hops": RAMP_HOPS,
                "robust_basin_hopping": ROBUST_BASIN_HOPPING,
                "angular_report_grid": {
                    "wavelengths_nm": [float(ANGULAR_WAVELENGTHS_NM[0]),
                                       float(ANGULAR_WAVELENGTHS_NM[-1]),
                                       len(ANGULAR_WAVELENGTHS_NM)],
                    "angles_deg": [float(ANGULAR_ANGLES_DEG[0]),
                                   float(ANGULAR_ANGLES_DEG[-1]), len(ANGULAR_ANGLES_DEG)]},
            },
            "materials": {
                "medium_substrate": glass.name,
                "high_index": HIGH_INDEX_MATERIAL.name,
                "high_n_at_reference": high_index,
                "low_index": LOW_INDEX_MATERIAL.name,
                "low_n_at_reference": low_index,
            },
            "seed": {
                "num_films": NUM_SEED_FILMS,
                "n_high": high_index, "n_low": low_index,
                "thickness_high_nm": seed_thickness_high,
                "thickness_low_nm": seed_thickness_low,
                "design_aoi_deg": aoi_design,
            },
            "thickness_bounds_nm": list(THICKNESS_BOUNDS_NM),
            "tolerances": {
                "thickness_plus_minus_fraction": (
                    THICKNESS_TOLERANCE_PLUS_MINUS_FRACTION),
                "thickness_plus_minus_percent": (
                    100.0 * THICKNESS_TOLERANCE_PLUS_MINUS_FRACTION),
                "atlas_sigma_definition": (
                    "one third of 1% of each nominal film thickness"),
                "n_sigma": N_TOLERANCE,
                "robust_optimization_samples": ROBUST_OPTIMIZATION_SAMPLES,
                "validation_samples": TOLERANCE_VALIDATION_SAMPLES,
                "validation_sampling": "scrambled Sobol",
                "validation_max_rs_error": TOLERANCE_MAX_RS_ERROR,
            },
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
    design_path.write_text(json.dumps(design, indent=2, default=float))

    print(f"\n── Done ({elapsed:.0f}s) ──")
    region_names = [region["name"] for region in SUPPRESSION_REGIONS]
    print(f"{'mirror':>6}  {'films':>5}  {'merit':>11}  {'target R':>12}  "
          f"{'display Rs':>13}  {'Rp max':>6}  "
          + "  ".join(f"{name + ' worst':>21}" for name in region_names)
          + f"  {'yield':>6}")
    for record in mirror_records:
        target_values = record["target_reflectance"]["values"]
        metrics = record["result"]["angular"]["metrics"]
        rs_low, rs_high = metrics["display"]["rs_weighted_range"]
        print(f"{record['index']:>6}  {record['result']['num_films']:>5}  "
              f"{record['result']['merit']:>11.3e}  "
              f"{min(target_values):.3f}..{max(target_values):.3f}  "
              f"{100 * rs_low:5.2f}–{100 * rs_high:5.2f}%  "
              f"{100 * metrics['display']['rp_max']:5.2f}%  "
              + "  ".join(f"{100 * metrics['regions'][name]['worst_unpolarized']:20.2f}%"
                          for name in region_names)
              + f"  {record['result']['tolerance_validation']['yield_fraction']:6.1%}")
    print(f"\nSaved consolidated design: {design_path}")
    if SAVE_PER_MIRROR_PLOTS:
        print(f"Per-mirror plots:          {out_dir}/coating_mirror_*.html")


if __name__ == "__main__":
    main()
