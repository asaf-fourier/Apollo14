"""Design a Perseus mirror coating that reflects the display and suppresses ghosts.

A multi-target **Atlas** design for one embedded partial mirror::

    glass | alternating HIGH / LOW films | glass        (index-matched, in N-BASF2)

All angles are angles of incidence **inside the glass**. The coating must:

- **display Rs** — reflect ``RS_TARGET`` (≈5%) of the s-polarized projector light
  over the **display AOI band**, computed from the Perseus geometry (the FOV
  refracted into the glass at the entry face, ≈46.9–52.7°). Sampled on the
  projector's emission bands and weighted by its spectrum;
- **display Rp** — keep p reflection low at the display angles (ambient light in
  the FOV is unpolarized, so low Rp means more see-through);
- **suppress** unpolarized reflection over 420–680 nm at ``SUPPRESSION_REGIONS``
  (default **10–36°** and **60–75°**). These come from the sun-ray ghost trace
  (``visualize_perseus_sun.py``): ghosts that TIR off a chassis face reach the
  eye via a last reflection at ~25–36°, after a first reflection at ~66–73°.
  The angles between those regions and the display band are guard bands —
  reflection cannot switch abruptly next to the display angles.

How strongly to suppress is a trade-off, not a known number: an unsuppressed
ghost region's loss (≈R² ≈ 0.25 at worst) dwarfs the display loss (~5e-5), so
even small suppression weights dominate. The script therefore fits the display
alone first (weight 0), then **log-sweeps** the suppression weight over
``SUPPRESSION_WEIGHTS`` (1e-4 … 1e-1), warm-starting each fit from the previous
design. It keeps the design with the lowest worst-case suppression reflectance
whose display error is at most ``DISPLAY_ERROR_ALLOWANCE`` worse than the
display-only fit — unless ``SELECTED_WEIGHT`` names a sweep weight to use
directly (currently 1e-2) — and reports every candidate so the whole trade-off
is visible.

Materials, glass, thickness bounds and the ±1% thickness tolerance follow
``design_perseus_mirror_coating.py``; the tolerance uses its
``_atlas_thickness_sigma_nm``. The sweep is nominal (non-robust); set
``ROBUST_FINAL = True`` to re-optimize the chosen design under the tolerance.

Outputs (``examples/reports/design_perseus_ghost_suppressed_coating/<timestamp>/``):

- ``ghost_suppressed_coating.json`` — spec, every candidate's metrics, the
  chosen layer recipe with deposition order, and its R(λ, θ) maps;
- ``ghost_suppressed_coating.html`` — unpolarized R(θ, λ) map, Rs/Rp/R vs AOI
  (projector-weighted), and the display-error vs suppression trade-off.

Run::

    python examples/design_perseus_ghost_suppressed_coating.py
"""

# Atlas must precede JAX here, so this intentionally differs from isort order.
# ruff: noqa: I001
import json
import time
from datetime import UTC, datetime
from pathlib import Path

from atlas import Layer, MultiTarget, OpticalDesigner, Target, example_output_dir
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from apollo14.materials import refractive_index
from apollo14.perseus import (
    PERSEUS_BEAM_HEIGHT,
    PERSEUS_BEAM_WIDTH,
    PERSEUS_LIGHT_POSITION,
    PERSEUS_PROJECTOR_DIRECTION,
)
from apollo14.projector import PlayNitrideLed
from apollo14.units import nm

from design_perseus_mirror_coating import (
    GLASS,
    HIGH_INDEX_MATERIAL,
    LOW_INDEX_MATERIAL,
    REFERENCE_WAVELENGTH_NM,
    THICKNESS_BOUNDS_NM,
    THICKNESS_TOLERANCE_PLUS_MINUS_FRACTION,
    _atlas_thickness_sigma_nm,
    _deposition_to_dict,
    _latest_report_dir,
    _layer_to_dict,
    _qwot_thickness_nm,
    incidence_angle_range,
    load_report,
)

# ── Spec ─────────────────────────────────────────────────────────────────────
RS_TARGET = 0.05                         # display s reflectance
BAND_WAVELENGTHS_NM = (420.0, 680.0)     # visible band for suppression / Rp
# Display wavelengths: where the projector emits at least this fraction of its
# peak radiance, widened by BAND_MARGIN_NM, sampled every DISPLAY_STEP_NM.
EMISSION_THRESHOLD = 0.10
BAND_MARGIN_NM = 10.0
DISPLAY_STEP_NM = 3.0
DISPLAY_NUM_ANGLES = 5
DISPLAY_RP_WEIGHT = 0.3

SUPPRESSION_REGIONS = [
    dict(name="suppress 10-36°", angles_deg=(10.0, 36.0), num_angles=14),
    dict(name="suppress 60-75°", angles_deg=(60.0, 75.0), num_angles=16),
]
SUPPRESSION_NUM_WAVELENGTHS = 40
SUPPRESSION_AGGREGATE = "soft_max"       # a ghost is as visible as its worst angle
SUPPRESSION_WEIGHTS = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1)
DISPLAY_ERROR_ALLOWANCE = 0.005          # 0.5 pp above the display-only fit
# Use this sweep weight's design; None picks automatically (lowest mean worst
# suppression R within the display-error allowance). 1e-2 was chosen from the
# 2026-11-10 sweep: best 10–36° suppression (the TIR ghosts' last reflection),
# lowest worst-R product over the two regions, and display Rs still ≈ 5%.
SELECTED_WEIGHT: float | None = 1e-2

# ── Stack and optimizer ──────────────────────────────────────────────────────
NUM_FILMS = 11
SEED_THICKNESS_HIGH_NM = 20.0            # thin high-index films: low-R starting basin
BH_HOPS = 8                              # basin-hopping hops per candidate
BH_LOCAL_MAXITER = 300
BH_STEPSIZE = 0.2
BH_TEMPERATURE = 0.01
RANDOM_SEED = 1
ROBUST_FINAL = False
ROBUST_SAMPLES = 32

# ── Evaluation / plots ───────────────────────────────────────────────────────
EVALUATION_WAVELENGTHS_NM = np.linspace(*BAND_WAVELENGTHS_NM, 131)
EVALUATION_ANGLES_DEG = np.arange(0.0, 85.5, 1.0)
OUTPUT_ROOT = "examples/reports"
# Validated categorical slots (light surface): Rs, Rp, unpolarized.
SERIES_COLORS = {"s": "#2a78d6", "p": "#eb6834", "both": "#1baf7a"}
SERIES_NAMES = {"s": "Rs", "p": "Rp", "both": "R unpolarized"}


# ── Spec construction ────────────────────────────────────────────────────────

def projector_spectrum():
    """(wavelengths_nm, radiance normalized to peak) of the Perseus projector."""
    projector = PlayNitrideLed.create_broadband(
        position=PERSEUS_LIGHT_POSITION, direction=PERSEUS_PROJECTOR_DIRECTION,
        beam_width=PERSEUS_BEAM_WIDTH, beam_height=PERSEUS_BEAM_HEIGHT, nx=1, ny=1)
    wavelengths, radiance = projector.spectrum
    radiance = np.asarray(radiance)
    return np.asarray(wavelengths) / float(nm), radiance / radiance.max()


def display_wavelength_points(spectrum_wavelengths_nm, spectrum_radiance):
    """Wavelength samples covering the projector's emission bands (+ margin)."""
    grid = np.arange(BAND_WAVELENGTHS_NM[0], BAND_WAVELENGTHS_NM[1] + 1e-9, 1.0)
    radiance = np.interp(grid, spectrum_wavelengths_nm, spectrum_radiance)
    emitting = radiance >= EMISSION_THRESHOLD
    widened = np.zeros_like(emitting)
    for center in grid[emitting]:
        widened |= np.abs(grid - center) <= BAND_MARGIN_NM
    sampled = grid[widened][::int(DISPLAY_STEP_NM)]
    return sampled, display_bands(grid, widened)


def display_bands(grid, mask):
    """Contiguous (start, end) nm ranges of a boolean mask over ``grid``."""
    bands, start, previous_wavelength = [], None, None
    for wavelength, inside in zip(grid, mask, strict=True):
        if inside and start is None:
            start = wavelength
        if not inside and start is not None:
            bands.append((float(start), float(previous_wavelength)))
            start = None
        previous_wavelength = wavelength
    if start is not None:
        bands.append((float(start), float(grid[-1])))
    return bands


def build_target(display_angles, display_points, display_weights, suppression_weight):
    """Display Rs (match) + display Rp (at most 0) + suppression (at most 0)."""
    regions = [
        Target(wavelength_points=display_points, angles=display_angles,
               num_angles=DISPLAY_NUM_ANGLES, polarization="s", reflectance=RS_TARGET,
               weights=display_weights, name="display Rs"),
        Target(wavelengths=BAND_WAVELENGTHS_NM, num_wavelengths=SUPPRESSION_NUM_WAVELENGTHS,
               angles=display_angles, num_angles=3, polarization="p", reflectance=0.0,
               goal="at_most", weight=DISPLAY_RP_WEIGHT, name="display Rp"),
    ]
    for region in SUPPRESSION_REGIONS:
        regions.append(Target(
            wavelengths=BAND_WAVELENGTHS_NM, num_wavelengths=SUPPRESSION_NUM_WAVELENGTHS,
            angles=region["angles_deg"], num_angles=region["num_angles"],
            polarization="both", reflectance=0.0, goal="at_most",
            aggregate=SUPPRESSION_AGGREGATE, weight=suppression_weight,
            name=region["name"]))
    return MultiTarget(regions=regions)


def seed_layers(glass, seed_thickness_low):
    """glass | HIGH (thin) / LOW (QWOT) alternating, H first | glass."""
    layers = [Layer(material=glass)]
    for film_idx in range(NUM_FILMS):
        is_high = film_idx % 2 == 0
        thickness = SEED_THICKNESS_HIGH_NM if is_high else seed_thickness_low
        layers.append(Layer(
            material=HIGH_INDEX_MATERIAL if is_high else LOW_INDEX_MATERIAL,
            thickness=float(np.clip(thickness, *THICKNESS_BOUNDS_NM)),
            vary_thickness=True, thickness_bounds=THICKNESS_BOUNDS_NM,
            thickness_tolerance=_atlas_thickness_sigma_nm(thickness)))
    layers.append(Layer(material=glass))
    return layers


def with_rebased_tolerance(layers):
    """Copy of ``layers`` with each film's ±1% sigma rebased on its thickness."""
    for layer in layers[1:-1]:
        layer.thickness_tolerance = _atlas_thickness_sigma_nm(layer.thickness)
    return layers


# ── Metrics ──────────────────────────────────────────────────────────────────

def reflectance_maps(designer, layers):
    """R(λ, θ) for s, p and unpolarized on the evaluation grid."""
    return {polarization: np.asarray(designer.evaluate(
                layers=layers, wavelengths=EVALUATION_WAVELENGTHS_NM,
                angles=EVALUATION_ANGLES_DEG, polarization=polarization))
            for polarization in ("s", "p", "both")}


def angle_mask(low_deg, high_deg):
    return (low_deg <= EVALUATION_ANGLES_DEG) & (high_deg >= EVALUATION_ANGLES_DEG)


def candidate_metrics(maps, display_angles, spectrum_wavelengths_nm, spectrum_radiance):
    """Display fit (projector-weighted) and suppression reflectance."""
    weights = np.interp(EVALUATION_WAVELENGTHS_NM, spectrum_wavelengths_nm, spectrum_radiance)
    display = angle_mask(*display_angles)
    display_rs = maps["s"][:, display]
    # Projector-weighted Rs per display angle: what the eye-box actually gets.
    weighted_rs = (weights @ display_rs) / weights.sum()
    metrics = dict(
        display_weighted_rs_mean=float(weighted_rs.mean()),
        display_weighted_rs_range=(float(weighted_rs.min()), float(weighted_rs.max())),
        display_error=float(np.max(np.abs(weighted_rs - RS_TARGET))),
        display_pointwise_worst=float(np.max(np.abs(display_rs[weights > EMISSION_THRESHOLD]
                                                    - RS_TARGET))),
        display_rp_max=float(maps["p"][:, display].max()),
        display_unpolarized_mean=float(maps["both"][:, display].mean()),
        regions={},
    )
    for region in SUPPRESSION_REGIONS:
        mask = angle_mask(*region["angles_deg"])
        metrics["regions"][region["name"]] = dict(
            worst_unpolarized=float(maps["both"][:, mask].max()),
            mean_unpolarized=float(maps["both"][:, mask].mean()),
            worst_s=float(maps["s"][:, mask].max()),
            worst_p=float(maps["p"][:, mask].max()))
    metrics["suppression_score"] = float(np.mean(
        [region["worst_unpolarized"] for region in metrics["regions"].values()]))
    return metrics


# ── Plots ────────────────────────────────────────────────────────────────────

def figure_for(chosen, candidates, display_angles, spectrum_wavelengths_nm,
               spectrum_radiance):
    maps = chosen["maps"]
    figure = make_subplots(
        rows=3, cols=1, vertical_spacing=0.09, row_heights=[0.4, 0.33, 0.27],
        subplot_titles=(
            "Unpolarized reflectance R(θ, λ) of the chosen coating",
            "Reflectance vs AOI (projector-weighted average over wavelength)",
            "Trade-off: display error vs worst suppressed reflectance"))

    # 1. Heatmap — sequential single hue (magnitude).
    figure.add_trace(go.Heatmap(
        x=EVALUATION_ANGLES_DEG, y=EVALUATION_WAVELENGTHS_NM, z=100 * maps["both"],
        colorscale=[[0, "#fcfcfb"], [0.25, "#9ec5f4"], [0.5, "#3987e5"],
                    [0.75, "#1c5cab"], [1, "#0d366b"]],
        colorbar=dict(title="R %", len=0.38, y=0.81),
        hovertemplate="AOI %{x:.0f}°<br>%{y:.0f} nm<br>R %{z:.2f}%<extra></extra>"),
        row=1, col=1)

    # 2. Rs / Rp / unpolarized vs AOI, projector-weighted.
    weights = np.interp(EVALUATION_WAVELENGTHS_NM, spectrum_wavelengths_nm, spectrum_radiance)
    for polarization in ("s", "p", "both"):
        curve = 100 * (weights @ maps[polarization]) / weights.sum()
        figure.add_trace(go.Scatter(
            x=EVALUATION_ANGLES_DEG, y=curve, mode="lines",
            line=dict(color=SERIES_COLORS[polarization], width=2),
            name=SERIES_NAMES[polarization],
            hovertemplate="AOI %{x:.0f}°<br>%{y:.2f}%<extra>"
                          + SERIES_NAMES[polarization] + "</extra>"),
            row=2, col=1)
        # Direct label at the line end (identity is never color-alone).
        figure.add_annotation(x=EVALUATION_ANGLES_DEG[-1], y=float(curve[-1]),
                              text=SERIES_NAMES[polarization], showarrow=False,
                              xanchor="left", font=dict(color="#52514e", size=11),
                              row=2, col=1)

    # Region shading on the first two panels: display band and suppression.
    for row in (1, 2):
        figure.add_vrect(x0=display_angles[0], x1=display_angles[1], row=row, col=1,
                         fillcolor="rgba(42,120,214,0.10)", line_width=0,
                         annotation_text="display" if row == 2 else None,
                         annotation_position="top left")
        for region in SUPPRESSION_REGIONS:
            figure.add_vrect(x0=region["angles_deg"][0], x1=region["angles_deg"][1],
                             row=row, col=1, fillcolor="rgba(82,81,78,0.08)",
                             line=dict(color="#52514e", width=1, dash="dot"),
                             annotation_text="suppress" if row == 2 else None,
                             annotation_position="top left")
    figure.add_hline(y=100 * RS_TARGET, row=2, col=1,
                     line=dict(color="#52514e", width=1, dash="dash"),
                     annotation_text=f"Rs target {100 * RS_TARGET:.1f}%",
                     annotation_position="bottom right")

    # 3. Trade-off: one dot per candidate, the chosen one ringed.
    figure.add_trace(go.Scatter(
        x=[100 * candidate["metrics"]["display_error"] for candidate in candidates],
        y=[100 * candidate["metrics"]["suppression_score"] for candidate in candidates],
        mode="lines+markers+text",
        text=[f"w={candidate['weight']:g}" for candidate in candidates],
        textposition="top right", textfont=dict(color="#52514e", size=10),
        marker=dict(size=9, color="#2a78d6", line=dict(color="#fcfcfb", width=2)),
        line=dict(color="#86b6ef", width=2), name="candidates", showlegend=False,
        hovertemplate=("display error %{x:.2f} pp<br>worst suppressed R %{y:.2f}%"
                       "<extra>%{text}</extra>")),
        row=3, col=1)
    figure.add_trace(go.Scatter(
        x=[100 * chosen["metrics"]["display_error"]],
        y=[100 * chosen["metrics"]["suppression_score"]], mode="markers",
        marker=dict(size=16, color="rgba(0,0,0,0)", line=dict(color="#0b0b0b", width=2)),
        name="chosen", hoverinfo="skip"), row=3, col=1)

    figure.update_xaxes(title_text="AOI in glass (°)", range=[0, 85], row=1, col=1)
    figure.update_yaxes(title_text="Wavelength (nm)", row=1, col=1)
    figure.update_xaxes(title_text="AOI in glass (°)", range=[0, 92], row=2, col=1)
    figure.update_yaxes(title_text="Reflectance (%)", rangemode="tozero", row=2, col=1)
    figure.update_xaxes(title_text="Display error, projector-weighted worst |Rs − target| (pp)",
                        row=3, col=1)
    figure.update_yaxes(title_text="Worst suppressed R (%)", rangemode="tozero", row=3, col=1)
    figure.update_layout(
        template="plotly_white", height=1250, hovermode="closest",
        paper_bgcolor="#fcfcfb", plot_bgcolor="#fcfcfb",
        font=dict(color="#0b0b0b"),
        title=(f"Perseus ghost-suppressed coating — Rs {100 * RS_TARGET:.1f}% at "
               f"{display_angles[0]:.1f}–{display_angles[1]:.1f}°, "
               f"suppression weight {chosen['weight']:g}"),
        legend=dict(x=1.0, y=0.55))
    return figure


# ── Run ──────────────────────────────────────────────────────────────────────

def main():
    report_dir = _latest_report_dir()
    _, num_mirrors, _, _, mirror_normal, _ = load_report(report_dir)
    glass_index = float(refractive_index(GLASS, REFERENCE_WAVELENGTH_NM * nm))
    aoi_min, aoi_max, aoi_on_axis, _ = incidence_angle_range(
        mirror_normal, num_mirrors, glass_index)
    display_angles = (aoi_min, aoi_max)

    spectrum_wavelengths_nm, spectrum_radiance = projector_spectrum()
    display_points, bands = display_wavelength_points(spectrum_wavelengths_nm,
                                                      spectrum_radiance)
    display_weights = (0.2 + 0.8 * np.interp(display_points, spectrum_wavelengths_nm,
                                             spectrum_radiance)).tolist()
    low_index = float(Layer(material=LOW_INDEX_MATERIAL, thickness=1.0)
                      .index_at(REFERENCE_WAVELENGTH_NM).real)
    seed_thickness_low = _qwot_thickness_nm(REFERENCE_WAVELENGTH_NM, low_index,
                                            glass_index, 0.5 * (aoi_min + aoi_max))

    print("── Perseus ghost-suppressed coating ──")
    print(f"geometry from : {report_dir}")
    print(f"glass         : {GLASS.name}, n={glass_index:.4f}")
    print(f"display band  : {aoi_min:.1f}–{aoi_max:.1f}° (on-axis {aoi_on_axis:.1f}°), "
          f"Rs target {100 * RS_TARGET:.1f}% s-pol")
    print("display λ     : " + ", ".join(f"{start:.0f}–{end:.0f} nm" for start, end in bands)
          + f" ({len(display_points)} samples)")
    print("suppress      : " + ", ".join(
        f"{region['angles_deg'][0]:.0f}–{region['angles_deg'][1]:.0f}°"
        for region in SUPPRESSION_REGIONS) + f", unpolarized, "
          f"{BAND_WAVELENGTHS_NM[0]:.0f}–{BAND_WAVELENGTHS_NM[1]:.0f} nm")
    print(f"stack         : {NUM_FILMS} × {HIGH_INDEX_MATERIAL.name}/{LOW_INDEX_MATERIAL.name}, "
          f"{THICKNESS_BOUNDS_NM[0]:.0f}–{THICKNESS_BOUNDS_NM[1]:.0f} nm, seed H "
          f"{SEED_THICKNESS_HIGH_NM:.0f} nm / L {seed_thickness_low:.0f} nm")
    print(f"weights       : 0 (display only), then {', '.join(f'{w:g}' for w in SUPPRESSION_WEIGHTS)}\n")

    run_start = time.perf_counter()
    layers = seed_layers(GLASS, seed_thickness_low)
    candidates = []
    for suppression_weight in (0.0, *SUPPRESSION_WEIGHTS):
        target = build_target(display_angles, display_points, display_weights,
                              suppression_weight)
        designer = OpticalDesigner(layers=layers, target=target)
        result = designer.optimize(
            method="basin_hopping", max_iterations=BH_HOPS, local_maxiter=BH_LOCAL_MAXITER,
            stepsize=BH_STEPSIZE, temperature=BH_TEMPERATURE, seed=RANDOM_SEED)
        layers = with_rebased_tolerance(result.layers)
        maps = reflectance_maps(designer, result.layers)
        metrics = candidate_metrics(maps, display_angles, spectrum_wavelengths_nm,
                                    spectrum_radiance)
        candidates.append(dict(weight=suppression_weight, designer=designer,
                               result=result, maps=maps, metrics=metrics))
        print(f"  weight {suppression_weight:<7g} display Rs "
              f"{100 * metrics['display_weighted_rs_range'][0]:.2f}–"
              f"{100 * metrics['display_weighted_rs_range'][1]:.2f}% "
              f"(error {100 * metrics['display_error']:.2f} pp), Rp max "
              f"{100 * metrics['display_rp_max']:.2f}% | "
              + ", ".join(f"{name}: worst {100 * values['worst_unpolarized']:.2f}% "
                          f"mean {100 * values['mean_unpolarized']:.2f}%"
                          for name, values in metrics["regions"].items()), flush=True)

    error_limit = candidates[0]["metrics"]["display_error"] + DISPLAY_ERROR_ALLOWANCE
    if SELECTED_WEIGHT is not None:
        matches = [candidate for candidate in candidates
                   if np.isclose(candidate["weight"], SELECTED_WEIGHT)]
        if not matches:
            raise ValueError(f"SELECTED_WEIGHT={SELECTED_WEIGHT:g} is not in the sweep "
                             f"(0, {', '.join(f'{w:g}' for w in SUPPRESSION_WEIGHTS)})")
        chosen = matches[0]
        print(f"\nchosen weight {chosen['weight']:g} (SELECTED_WEIGHT; display error "
              f"{100 * chosen['metrics']['display_error']:.2f} pp vs automatic limit "
              f"{100 * error_limit:.2f} pp)")
    else:
        eligible = [candidate for candidate in candidates
                    if candidate["metrics"]["display_error"] <= error_limit]
        chosen = min(eligible, key=lambda candidate: candidate["metrics"]["suppression_score"])
        print(f"\nchosen weight {chosen['weight']:g} "
              f"(display error limit {100 * error_limit:.2f} pp = display-only "
              f"{100 * candidates[0]['metrics']['display_error']:.2f} pp + "
              f"{100 * DISPLAY_ERROR_ALLOWANCE:.2f} pp allowance)")

    if ROBUST_FINAL:
        print("re-optimizing the chosen design under the ±1% thickness tolerance…")
        robust_designer = OpticalDesigner(layers=with_rebased_tolerance(chosen["result"].layers),
                                          target=chosen["designer"].target)
        robust_result = robust_designer.optimize(method="lbfgs", max_iterations=300,
                                                 robust=True, num_samples=ROBUST_SAMPLES)
        maps = reflectance_maps(robust_designer, robust_result.layers)
        chosen = dict(chosen, designer=robust_designer, result=robust_result, maps=maps,
                      metrics=candidate_metrics(maps, display_angles,
                                                spectrum_wavelengths_nm, spectrum_radiance),
                      robust=True)

    elapsed = time.perf_counter() - run_start
    film_layers = with_rebased_tolerance(chosen["result"].layers)[1:-1]
    out_dir = Path(example_output_dir(__file__, root=OUTPUT_ROOT))
    out_dir.mkdir(parents=True, exist_ok=True)

    design = {
        "generated_utc": datetime.now(UTC).isoformat(),
        "wall_time_s": round(elapsed, 1),
        "source_report": str(report_dir),
        "spec": {
            "rs_target": RS_TARGET,
            "glass": {"name": GLASS.name, "index": glass_index,
                      "reference_wavelength_nm": REFERENCE_WAVELENGTH_NM},
            "display_angles_deg": list(display_angles),
            "display_on_axis_deg": aoi_on_axis,
            "display_wavelength_bands_nm": bands,
            "display_rp_weight": DISPLAY_RP_WEIGHT,
            "suppression_regions": [dict(name=region["name"],
                                         angles_deg=list(region["angles_deg"]),
                                         polarization="both",
                                         wavelengths_nm=list(BAND_WAVELENGTHS_NM),
                                         goal="at_most 0", aggregate=SUPPRESSION_AGGREGATE)
                                    for region in SUPPRESSION_REGIONS],
            "suppression_weights": [0.0, *SUPPRESSION_WEIGHTS],
            "display_error_allowance": DISPLAY_ERROR_ALLOWANCE,
            "selected_weight": SELECTED_WEIGHT,
            "num_films": NUM_FILMS,
            "materials": {"high": HIGH_INDEX_MATERIAL.name, "low": LOW_INDEX_MATERIAL.name},
            "thickness_bounds_nm": list(THICKNESS_BOUNDS_NM),
            "thickness_tolerance_plus_minus_percent":
                100.0 * THICKNESS_TOLERANCE_PLUS_MINUS_FRACTION,
            "robust_final": ROBUST_FINAL,
        },
        "candidates": [dict(weight=candidate["weight"], metrics=candidate["metrics"],
                            merit=float(candidate["result"].merit),
                            thickness_nm=[float(layer.thickness)
                                          for layer in candidate["result"].film_layers])
                       for candidate in candidates],
        "chosen": {
            "weight": chosen["weight"],
            "metrics": chosen["metrics"],
            "regions": [region.to_dict() for region in chosen["result"].regions],
            "layers": [_layer_to_dict(layer) for layer in film_layers],
            "deposition": _deposition_to_dict(film_layers),
            "reflectance_maps": {
                "wavelengths_nm": EVALUATION_WAVELENGTHS_NM.tolist(),
                "angles_deg": EVALUATION_ANGLES_DEG.tolist(),
                **{f"R_{polarization}": chosen["maps"][polarization].tolist()
                   for polarization in ("s", "p", "both")},
            },
        },
    }
    json_path = out_dir / "ghost_suppressed_coating.json"
    json_path.write_text(json.dumps(design, indent=2, default=float))
    html_path = out_dir / "ghost_suppressed_coating.html"
    figure_for(chosen, candidates, display_angles, spectrum_wavelengths_nm,
               spectrum_radiance).write_html(str(html_path), include_plotlyjs="cdn")

    metrics = chosen["metrics"]
    print(f"\n── Chosen design (weight {chosen['weight']:g}, {elapsed:.0f}s) ──")
    print(f"display Rs (projector-weighted, over FOV): "
          f"{100 * metrics['display_weighted_rs_range'][0]:.2f}–"
          f"{100 * metrics['display_weighted_rs_range'][1]:.2f}%  "
          f"(pointwise worst |Rs−target| {100 * metrics['display_pointwise_worst']:.2f} pp)")
    print(f"display Rp max {100 * metrics['display_rp_max']:.2f}%, unpolarized mean "
          f"{100 * metrics['display_unpolarized_mean']:.2f}%")
    for name, values in metrics["regions"].items():
        print(f"{name}: unpolarized worst {100 * values['worst_unpolarized']:.2f}% / "
              f"mean {100 * values['mean_unpolarized']:.2f}%  "
              f"(Rs worst {100 * values['worst_s']:.2f}%, Rp worst {100 * values['worst_p']:.2f}%)")
    print("layers (layers[0] faces the incident glass):")
    for layer_idx, layer in enumerate(film_layers):
        print(f"  {layer_idx:2d}. {layer.material.name:15s} {layer.thickness:7.2f} nm")
    print(f"\nSaved: {json_path}\n       {html_path}")


if __name__ == "__main__":
    main()
