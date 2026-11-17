"""Perseus visualization — the exact system ``optimize_pupil_perseus.py`` optimizes.

The scene is built by the optimizer's own :func:`build_system` (which wraps
:func:`helios.perseus_params.build_parametrized_perseus` with the optimizer's
glass, chassis depth and wavelength grid), and the beam comes from the
optimizer's own ``PROJECTOR`` — so geometry, glass, AR coating, aperture,
mirrors, pupil, beam size and ray-placement mode are always identical to what
is being optimized. Nothing about the system is configured here; change
``apollo14/perseus.py`` or ``optimize_pupil_perseus.py`` and this follows.

Mirror reflectances (``REFLECTANCE_SOURCE``):

- ``"latest_report"`` (default) — the final design of the newest
  ``optimize_pupil_perseus`` run, used only if that run's saved system matches
  the current geometry element by element (mirror placement and size, chassis
  glass); otherwise it falls back to the optimizer's starting design and says
  why;
- ``"initial"`` — the optimizer's warm-start (Talos cascade-compensated) design.

The beam is ray-traced along the optimizer's routes — the main transmitted
path plus one reflected branch per mirror onto the pupil — and the projector
scans the optimizer's FOV, with a slider to step through scan angles. Only the
*drawing* density is chosen here (``DRAW_RAYS_*``, ``SCAN_STEPS_*``): the
drawn beam is the optimizer's projector with fewer rays, so it has the same
position, direction, cross-section and sampling mode.

Run::

    python examples/visualize_perseus.py
"""

import dataclasses
import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import plotly.graph_objects as go

sys.path.insert(0, str(Path(__file__).parent))
import optimize_pupil_perseus as optimizer  # the system's single source

from apollo14.elements.aperture import RectangularAperture
from apollo14.elements.glass_block import GlassBlock
from apollo14.elements.partial_mirror import PartialMirror
from apollo14.elements.pupil import RectangularPupil
from apollo14.projector import FovGrid
from apollo14.route import combiner_main_path
from apollo14.spectral import ConstantCurve, SumOfGaussiansCurve
from apollo14.trace import prepare_route, trace_rays
from apollo14.units import deg, mm, nm
from apollo14.visualizer import eyebox_overlay_traces, plot_system
from helios.combiner_params import CombinerParams
from helios.merit import build_combiner_branch_routes

# One JIT compile per distinct route structure, reused across scan directions.
_trace_rays_jit = jax.jit(trace_rays)

# ── Drawing choices only (the system itself comes from the optimizer) ───────
REFLECTANCE_SOURCE = "latest_report"   # "latest_report" | "initial"
DRAW_RAYS_X = 9                        # drawn rays across the beam width
DRAW_RAYS_Y = 5                        # drawn rays across the beam height
SCAN_STEPS_X = 5                       # slider steps across the optimizer's FOV
SCAN_STEPS_Y = 5
NORMAL_DRAW_LENGTH = 2.0 * mm          # length of each drawn normal
REPORT_PATH = Path("examples/reports/visualize_perseus.html")


# ── Which design to show ─────────────────────────────────────────────────────

def _params_from_report(report: dict) -> CombinerParams:
    """Rebuild the optimizer's final ``CombinerParams`` from a saved report."""
    final = report["final_params"]
    curve = final["curve"]
    amplitude = jnp.asarray(curve["amplitude"], dtype=jnp.float32)
    if curve["type"] == "ConstantCurve":
        curves = ConstantCurve(amplitude=amplitude)
    elif curve["type"] == "SumOfGaussiansCurve":
        curves = SumOfGaussiansCurve(
            amplitude=amplitude,
            sigma=jnp.asarray(curve["sigma_nm"]) * nm,
            centers=jnp.asarray(curve["centers_nm"]) * nm)
    else:
        raise ValueError(f"unsupported curve type {curve['type']!r}")
    return CombinerParams(spacings=jnp.asarray(final["spacings"]) * mm, curves=curves)


def _geometry_mismatches(saved_system: dict, current_system) -> list[str]:
    """Differences between a report's saved system and the current geometry."""
    saved_mirrors = [element for element in saved_system["elements"]
                     if element["type"] == "PartialMirror"]
    current_mirrors = [element for element in current_system.elements
                       if isinstance(element, PartialMirror)]
    if len(saved_mirrors) != len(current_mirrors):
        return [f"{len(saved_mirrors)} mirrors saved vs {len(current_mirrors)} now"]
    problems = []
    for saved, current in zip(saved_mirrors, current_mirrors, strict=True):
        for field in ("position", "normal"):
            if not np.allclose(saved[field], np.asarray(getattr(current, field)), atol=1e-3):
                problems.append(f"{saved['name']} {field} differs")
        for field in ("width", "height"):
            if not np.isclose(saved[field], float(getattr(current, field)), atol=1e-3):
                problems.append(f"{saved['name']} {field} {saved[field]:.3f} vs "
                                f"{float(getattr(current, field)):.3f}")
    saved_glass = [element.get("material") for element in saved_system["elements"]
                   if element["type"] == "GlassBlock"]
    current_glass = [element.material.name for element in current_system.elements
                     if isinstance(element, GlassBlock)]
    if [name.lower() for name in saved_glass] != [name.lower() for name in current_glass]:
        problems.append(f"chassis glass {saved_glass} vs {current_glass}")
    return problems


def design_to_show() -> tuple[CombinerParams, str]:
    """(params, description) per ``REFLECTANCE_SOURCE``, verified against geometry."""
    initial = optimizer._build_initial_params()
    if REFLECTANCE_SOURCE == "initial":
        return initial, "optimizer warm start (REFLECTANCE_SOURCE='initial')"
    reports = sorted(optimizer.RUNS_ROOT.glob("*/optimization_report.json"))
    if not reports:
        return initial, "optimizer warm start (no optimize_pupil_perseus report yet)"
    report = json.loads(reports[-1].read_text())
    run_name = reports[-1].parent.name
    if report["optimizer"].get("curve_mode") != optimizer.CURVE_MODE:
        return initial, (f"optimizer warm start (report {run_name} is "
                         f"{report['optimizer'].get('curve_mode')!r}, optimizer is "
                         f"{optimizer.CURVE_MODE!r})")
    params = _params_from_report(report)
    problems = _geometry_mismatches(report["system"], optimizer.build_system(params))
    if problems:
        return initial, (f"optimizer warm start (report {run_name} was optimized on a "
                         f"different geometry: {'; '.join(problems[:3])}"
                         f"{' …' if len(problems) > 3 else ''})")
    return params, f"final design of optimizer run {run_name}"


# ── Tracing ──────────────────────────────────────────────────────────────────

def build_routes(system, wavelength) -> list:
    """The optimizer's routes: main transmitted path + one branch per mirror."""
    main_route = prepare_route(combiner_main_path(system), wavelength)
    branch_routes = [prepare_route(route, wavelength)
                     for route in build_combiner_branch_routes(
                         system, num_mirrors=optimizer.NUM_MIRRORS)]
    return [main_route, *branch_routes]


def trace_scan(routes, projector, wavelength, fov_grid) -> list:
    """All routes at every scan direction, angle-major (``plot_system`` order)."""
    results = []
    for direction in fov_grid.flat_directions:
        beam = projector.generate_rays(direction=direction, wavelength=wavelength)
        for route in routes:
            results.append(_trace_rays_jit(route, beam))
    return results


# ── Drawing helpers ──────────────────────────────────────────────────────────

def _unit(vector) -> np.ndarray:
    vector = np.asarray(vector, dtype=float)
    return vector / np.linalg.norm(vector)


def eye_face_normal(chassis: GlassBlock) -> np.ndarray:
    """Outward normal of the chassis's eye-side slab face (the combiner normal)."""
    center = np.mean([np.asarray(face.position) for face in chassis.faces], axis=0)
    outward = []
    for face in chassis.faces:
        normal = _unit(face.normal)
        if np.dot(np.asarray(face.position) - center, normal) < 0:
            normal = -normal
        outward.append(normal)
    return max(outward, key=lambda normal: normal[2])


def normal_trace(position, direction, name, color) -> go.Scatter3d:
    """A short line along ``direction`` from ``position`` (an element normal)."""
    start = np.asarray(position)
    end = start + NORMAL_DRAW_LENGTH * _unit(direction)
    return go.Scatter3d(
        x=[float(start[0]), float(end[0])], y=[float(start[1]), float(end[1])],
        z=[float(start[2]), float(end[2])], mode="lines",
        line=dict(color=color, width=6), name=name, hoverinfo="name")


def angle_arc_traces(apex, dir_a, dir_b, *, radius=1.3 * mm,
                     line_length=2.0 * mm, color="magenta",
                     num_points=48) -> list:
    """Arc + label marking the angle between ``dir_a`` and ``dir_b`` at ``apex``."""
    apex = np.asarray(apex, dtype=float)
    unit_a, unit_b = _unit(dir_a), _unit(dir_b)
    theta = float(np.arccos(np.clip(np.dot(unit_a, unit_b), -1.0, 1.0)))
    fractions = np.linspace(0.0, 1.0, num_points)[:, None]
    arc = apex[None, :] + radius * (np.sin((1.0 - fractions) * theta) * unit_a
                                    + np.sin(fractions * theta) * unit_b) / np.sin(theta)

    def _radial(direction, **line_kw):
        end = apex + line_length * direction
        return go.Scatter3d(
            x=[apex[0], end[0]], y=[apex[1], end[1]], z=[apex[2], end[2]],
            mode="lines", line=line_kw, hoverinfo="skip", showlegend=False)

    label_pos = apex + radius * 1.45 * _unit(unit_a + unit_b)
    return [
        _radial(unit_a, color=color, width=3),
        _radial(unit_b, color=color, width=3, dash="dot"),
        go.Scatter3d(x=arc[:, 0], y=arc[:, 1], z=arc[:, 2], mode="lines",
                     line=dict(color=color, width=5),
                     name="mirror↔combiner normal angle", hoverinfo="name"),
        go.Scatter3d(x=[label_pos[0]], y=[label_pos[1]], z=[label_pos[2]],
                     mode="text", text=[f"{np.degrees(theta):.1f}°"],
                     textfont=dict(size=15, color=color),
                     hoverinfo="skip", showlegend=False),
    ]


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    params, design_description = design_to_show()
    system = optimizer.build_system(params)
    chassis = next(element for element in system.elements if isinstance(element, GlassBlock))
    aperture = next((element for element in system.elements
                     if isinstance(element, RectangularAperture)), None)
    mirrors = [element for element in system.elements if isinstance(element, PartialMirror)]
    pupil = next(element for element in system.elements if isinstance(element, RectangularPupil))
    projector = dataclasses.replace(optimizer.PROJECTOR, nx=DRAW_RAYS_X, ny=DRAW_RAYS_Y)
    combiner_normal = eye_face_normal(chassis)

    spec_wls, spec_rad = projector.spectrum
    trace_wavelength = float(spec_wls[int(jnp.argmax(spec_rad))])

    print("── Perseus — the system optimize_pupil_perseus.py optimizes ──")
    print(f"design     : {design_description}")
    print(f"projector  : {np.round(np.asarray(projector.position) / mm, 3)} mm, "
          f"direction {np.round(np.asarray(projector.direction), 4)}, beam "
          f"{projector.beam_width / mm:.1f} x {projector.beam_height / mm:.1f} mm "
          f"({'cell-centred' if projector.cell_centered else 'endpoint'} sampling; "
          f"optimizer {optimizer.PROJECTOR.nx}×{optimizer.PROJECTOR.ny} rays, "
          f"drawn {DRAW_RAYS_X}×{DRAW_RAYS_Y})")
    if aperture is not None:
        print(f"aperture   : {np.round(np.asarray(aperture.position) / mm, 3)} mm, frame "
              f"{aperture.width / mm:.1f} x {aperture.height / mm:.1f} mm, opening "
              f"{aperture.inner_width / mm:.1f} x {aperture.inner_height / mm:.1f} mm")
    print(f"chassis    : {chassis.material.name}, AR "
          f"{100 * float(np.mean(np.asarray(chassis.faces[0].coating_reflectance.values))):.2f}% "
          f"per face, eye-face normal {np.round(combiner_normal, 4)}")
    print(f"mirrors    : {len(mirrors)}, {float(mirrors[0].width) / mm:.1f} x "
          f"{float(mirrors[0].height) / mm:.2f} mm, normal {np.round(np.asarray(mirrors[0].normal), 4)}")
    print("reflectance: " + ", ".join(
        f"{100 * float(np.mean(np.asarray(mirror.reflectance))):.2f}%" for mirror in mirrors))
    print(f"pupil      : centre {np.round(np.asarray(pupil.position) / mm, 2)} mm, "
          f"{pupil.width / mm:.1f} x {pupil.height / mm:.1f} mm; eyebox "
          f"{2 * optimizer.EYEBOX_HALF_X / mm:.0f} x {2 * optimizer.EYEBOX_HALF_Y / mm:.0f} mm")

    routes = build_routes(system, trace_wavelength)
    fov_grid = FovGrid(optimizer.PROJECTOR_DIRECTION, optimizer.X_FOV, optimizer.Y_FOV,
                       num_x=SCAN_STEPS_X, num_y=SCAN_STEPS_Y)
    trace_results = trace_scan(routes, projector, trace_wavelength, fov_grid)
    rays_per_beam = DRAW_RAYS_X * DRAW_RAYS_Y
    branch_results = [result for index, result in enumerate(trace_results)
                      if index % len(routes) != 0]
    pupil_rays = sum(int(np.asarray(result.valids[:, -1]).sum()) for result in branch_results)
    print(f"ray trace  : {trace_wavelength / nm:.0f} nm, {SCAN_STEPS_X} x {SCAN_STEPS_Y} angles "
          f"over the optimizer FOV (±{optimizer.X_FOV / deg / 2:.1f}° × "
          f"±{optimizer.Y_FOV / deg / 2:.1f}°); branch rays on the pupil "
          f"{pupil_rays} / {len(fov_grid) * len(mirrors) * rays_per_beam}")

    figure = plot_system(system, trace_results=trace_results,
                         scan_angles=np.asarray(fov_grid.angles_grid),
                         projector=projector, show=False)
    figure.add_trace(normal_trace(projector.position, projector.direction,
                                  "projector normal", "red"))
    if aperture is not None:
        figure.add_trace(normal_trace(aperture.position, aperture.normal,
                                      "aperture normal", "orange"))
    stack_center = np.mean([np.asarray(mirror.position) for mirror in mirrors], axis=0)
    figure.add_trace(normal_trace(stack_center, combiner_normal, "combiner normal", "lime"))

    # Angle between the 4th mirror's normal and the combiner normal (all
    # mirrors share it), drawn at the 4th mirror.
    fourth_mirror = mirrors[min(3, len(mirrors) - 1)]
    print(f"mirror normal vs combiner normal: "
          f"{np.degrees(np.arccos(np.clip(abs(np.dot(_unit(fourth_mirror.normal), combiner_normal)), -1, 1))):.3f} deg")
    figure.add_traces(angle_arc_traces(fourth_mirror.position, fourth_mirror.normal,
                                       combiner_normal))

    for overlay in eyebox_overlay_traces(
            np.asarray(pupil.position),
            float(optimizer.EYEBOX_HALF_X), float(optimizer.EYEBOX_HALF_Y),
            np.asarray(pupil.normal),
            box_label=(f"target eyebox ({2 * optimizer.EYEBOX_HALF_X / mm:.0f}x"
                       f"{2 * optimizer.EYEBOX_HALF_Y / mm:.0f} mm)"),
            marker_label="pupil center"):
        figure.add_trace(overlay)

    figure.update_layout(title=f"Perseus — ray trace ({design_description})")
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(str(REPORT_PATH))
    print(f"\nSaved: {REPORT_PATH}")
    figure.show()


if __name__ == "__main__":
    main()
