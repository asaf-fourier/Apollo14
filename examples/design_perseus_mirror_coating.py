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

# Atlas must precede JAX here, so this intentionally differs from isort order.
# ruff: noqa: I001
import json
import time
from datetime import UTC, datetime
from pathlib import Path

# Import atlas BEFORE jax so its XLA multi-core flag takes effect (see
# atlas/__init__.py). apollo14 (which imports jax) is imported after.
from atlas import Layer, Materials, OpticalDesigner, Target, example_output_dir
from atlas.core.stack import Stack
from atlas.core.tmm import _precompute_fixed_nk_data, compute_rt_polarized
import jax
import jax.numpy as jnp
import numpy as np
import plotly.graph_objects as go
from plotly.colors import sample_colorscale
from scipy.stats import norm, qmc

from apollo14.geometry import snell_refract
from apollo14.materials import agc_m074
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
NUM_SEED_FILMS = 15
THICKNESS_BOUNDS_NM = (20.0, 200.0)
TIO2_N_BOUNDS = (2.0, 2.525)        # PLD_TiO2 tunable index range
AL2O3_N_BOUNDS = (1.47, 1.65)       # PLD_Al2O3 tunable index range
SEED_N_HIGH = 2.50                  # QWOT seed indices (max Δn)
SEED_N_LOW = 1.47
# Manufacturing thickness (height) tolerance is a hard ±2 nm specification.
# Atlas expresses tolerances as Gaussian 1σ values, so map the hard limit to
# three sigma for any future robust-optimization or tolerance analysis.
THICKNESS_TOLERANCE_PLUS_MINUS_NM = 2.0
ATLAS_THICKNESS_TOLERANCE_SIGMA_NM = THICKNESS_TOLERANCE_PLUS_MINUS_NM / 3.0
N_TOLERANCE = 0.005
ROBUST_OPTIMIZATION_SAMPLES = 256
TOLERANCE_VALIDATION_SAMPLES = 4096
TOLERANCE_VALIDATION_BATCH_SIZE = 256
TOLERANCE_MAX_RS_ERROR = 0.01
TOLERANCE_RANDOM_SEED = 42

# Optimizer budget PER MIRROR. Modest defaults so a full stack runs in minutes;
# bump for production recipes (BH max_iterations→100+, local_maxiter→500).
BH_MAX_ITERATIONS = 3
BH_LOCAL_MAXITER = 400
BH_STEPSIZE = 0.2
BH_TEMPERATURE = 0.01
POLISH_MAX_ITERATIONS = 500

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
            thickness_tolerance=ATLAS_THICKNESS_TOLERANCE_SIGMA_NM,
            n_tolerance=N_TOLERANCE,
            n_bounds=TIO2_N_BOUNDS if is_high else AL2O3_N_BOUNDS,
        ))
    layers.append(Layer(material=glass))                   # substrate = glass
    return layers

def _seed_moveon_layers(moveon, seed_thickness_high, seed_thickness_low) -> list:
    """moveon materials only."""
    layers = [Layer(material=moveon.MR10)]
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
            thickness_tolerance=ATLAS_THICKNESS_TOLERANCE_SIGMA_NM,
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
        layers=_seed_layers(glass, seed_thickness_high, seed_thickness_low),
        # layers=_seed_moveon_layers(Materials.moveon, seed_thickness_high, seed_thickness_low),
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
        method="basin_hopping", max_iterations=BH_MAX_ITERATIONS,
        robust=True, num_samples=ROBUST_OPTIMIZATION_SAMPLES,
        stepsize=BH_STEPSIZE, temperature=BH_TEMPERATURE,
        local_maxiter=BH_LOCAL_MAXITER, callback=hop_progress)

    refined = [result_bh.layers[0], *result_bh.film_layers, result_bh.layers[-1]]
    polished_designer = OpticalDesigner(layers=refined, target=target)
    result_polished = polished_designer.optimize(
        method="lbfgs", max_iterations=POLISH_MAX_ITERATIONS,
        robust=True, num_samples=ROBUST_OPTIMIZATION_SAMPLES)
    result = result_polished if result_polished.merit <= result_bh.merit else result_bh
    result.selected_stage = "polish" if result is result_polished else "basin_hopping"
    result.basin_merit = float(result_bh.merit)
    result.polish_merit = float(result_polished.merit)
    if result is result_bh:
        print(f"    polish merit {result_polished.merit:.4e} is worse than "
              f"basin best {result_bh.merit:.4e}; retaining basin result")

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


def save_mirror_plot_html(result, path: Path, config_text: str) -> None:
    """Interactive target-vs-achieved plot for one mirror, as a Plotly HTML file.

    Replaces Atlas's ``plot_per_angle`` PNG. These coatings sit near R≈3%, so a
    static image over a fixed 0–1 axis buries the whole result on the axis line;
    here the axes autoscale to the data and you can zoom into the per-angle
    spread and the ripple between the projector's emission bands.
    """
    wavelengths_nm = np.asarray(result.wavelengths_nm)
    target = np.asarray(result.target_reflectance)
    achieved_s = np.asarray(result.reflectance)
    achieved_p = np.asarray(result.p_reflectance)
    if achieved_s.ndim == 1:
        achieved_s = achieved_s[:, np.newaxis]
        achieved_p = achieved_p[:, np.newaxis]
    angles_deg = np.asarray(result.angles_deg)

    figure = go.Figure()
    # One trace per angle of incidence, coloured across the swept range.
    colors = sample_colorscale(
        "Viridis",
        np.linspace(0.1, 0.9, achieved_s.shape[1]).tolist())
    for angle_idx, (angle, color) in enumerate(
            zip(angles_deg, colors, strict=True)):
        figure.add_trace(go.Scatter(
            x=wavelengths_nm, y=achieved_s[:, angle_idx],
            mode="lines", name=f"S · {float(angle):.1f}°",
            line=dict(color=color, width=1.5),
            hovertemplate="%{x:.1f} nm<br>R=%{y:.4f}<extra>%{fullData.name}</extra>"))
        figure.add_trace(go.Scatter(
            x=wavelengths_nm, y=achieved_p[:, angle_idx],
            mode="lines", name=f"P · {float(angle):.1f}°",
            line=dict(color=color, width=1.5, dash="dot"),
            hovertemplate="%{x:.1f} nm<br>R=%{y:.4f}<extra>%{fullData.name}</extra>"))
    # Target last so it draws on top of the achieved family.
    figure.add_trace(go.Scatter(
        x=wavelengths_nm, y=target, mode="lines", name="S target",
        line=dict(color="black", width=2.5, dash="dash"),
        hovertemplate="%{x:.1f} nm<br>target R=%{y:.4f}<extra>Target</extra>"))

    if len(angles_deg) > 1:
        angle_span = (f"{float(angles_deg[0]):.1f}°–{float(angles_deg[-1]):.1f}° "
                      f"({len(angles_deg)} samples)")
    else:
        angle_span = f"{float(angles_deg[0]):.1f}°"

    figure.update_layout(
        title=(f"S/P reflectance vs target — angles {angle_span}, "
               f"merit={result.merit:.3e}"),
        xaxis_title="Wavelength (nm)",
        yaxis_title="Reflectance",
        hovermode="x unified",
        legend=dict(title="Angle"),
        template="plotly_white",
    )
    # Free y-axis: the point of the HTML version is that the reader zooms
    # instead of the author guessing a range.
    figure.update_yaxes(rangemode="normal", tickformat=".3f")
    figure.add_annotation(
        text=config_text.replace("\n", "<br>"),
        xref="paper", yref="paper", x=0.01, y=0.99,
        showarrow=False, align="left",
        font=dict(family="monospace", size=11),
        bgcolor="rgba(255,255,255,0.85)", bordercolor="#bbbbbb", borderwidth=1)

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


def validate_manufacturing_tolerance(result, *, seed: int) -> dict:
    """Evaluate bounded thickness and Gaussian index manufacturing errors."""
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
    thickness_error = (
        2.0 * unit_samples[:, :num_thicknesses] - 1.0
    ) * THICKNESS_TOLERANCE_PLUS_MINUS_NM
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

    def evaluate_one(thicknesses, indices):
        reflectance, _ = compute_rt_polarized(
            stack, wavelengths, angles, thicknesses, indices, fixed_nk_data)
        return reflectance

    evaluate_batch = jax.jit(jax.vmap(evaluate_one))
    sample_rmse = []
    sample_max_error = []
    sample_p_max = []
    for start in range(0, TOLERANCE_VALIDATION_SAMPLES,
                       TOLERANCE_VALIDATION_BATCH_SIZE):
        stop = min(start + TOLERANCE_VALIDATION_BATCH_SIZE,
                   TOLERANCE_VALIDATION_SAMPLES)
        reflectance = np.asarray(evaluate_batch(
            jnp.asarray(thickness_samples[start:stop]),
            jnp.asarray(index_samples[start:stop])))
        s_error = reflectance[..., 0] - target[None, :, :]
        sample_rmse.append(np.sqrt(np.mean(np.square(s_error), axis=(1, 2))))
        sample_max_error.append(np.max(np.abs(s_error), axis=(1, 2)))
        sample_p_max.append(np.max(reflectance[..., 1], axis=(1, 2)))

    sample_rmse = np.concatenate(sample_rmse)
    sample_max_error = np.concatenate(sample_max_error)
    sample_p_max = np.concatenate(sample_p_max)
    percentiles = [50, 90, 95, 99, 100]

    return {
        "samples": TOLERANCE_VALIDATION_SAMPLES,
        "sampling": "scrambled Sobol",
        "seed": seed,
        "thickness_error_distribution": "uniform bounded",
        "thickness_plus_minus_nm": THICKNESS_TOLERANCE_PLUS_MINUS_NM,
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
        validation = _validation_metrics(result)
        print(f"    → merit {result.merit:.4e}, {len(result.film_layers)} films, "
              f"Rs RMSE={validation['s_target_rmse']:.4e}, "
              f"worst={validation['s_target_max_abs_error']:.4e}, "
              f"Rp mean/max={validation['p_mean_reflectance']:.4f}/"
              f"{validation['p_max_reflectance']:.4f}")
        print(f"    tolerance validation: {TOLERANCE_VALIDATION_SAMPLES} samples")
        result.tolerance_validation = validate_manufacturing_tolerance(
            result, seed=TOLERANCE_RANDOM_SEED + mirror_index)
        print(f"    → tolerance yield="
              f"{result.tolerance_validation['yield_fraction']:.1%} "
              f"at max |Rs-target| ≤ {TOLERANCE_MAX_RS_ERROR:.3f}")

        if SAVE_PER_MIRROR_PLOTS:
            save_mirror_plot_html(
                result,
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
            "tolerances": {
                "thickness_plus_minus_nm": THICKNESS_TOLERANCE_PLUS_MINUS_NM,
                "atlas_thickness_sigma_nm": ATLAS_THICKNESS_TOLERANCE_SIGMA_NM,
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
        print(f"Per-mirror plots:          {out_dir}/coating_mirror_*.html")


if __name__ == "__main__":
    main()
