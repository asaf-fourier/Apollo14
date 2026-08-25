"""3D visualization of the flat combiner — the system optimized by
:mod:`examples.optimize_pupil_flat`.

This example does no optimization. It rebuilds the *exact same geometry*
that ``optimize_pupil_flat.py`` designs — 8 wavelength-flat partial
mirrors at a fixed 0.93 mm spacing inside a thin 1.7 mm glass chassis,
lit by the broadband-white PlayNitride projector — and renders it with
:func:`apollo14.visualizer.plot_system`:

- the glass chassis, the 8 gold partial mirrors (with normals), the
  beam-defining aperture, the exit pupil, and the projector box;
- the traced rays, colored per scan angle, with a slider that steps
  through the FOV so you can watch where light lands as the eye gazes
  around;
- a cyan wireframe overlay of the 8x10 mm eyebox on the pupil plane.

Every knob of the visualized system is collected in one place:
:class:`SystemConfig` — a declarative spec sheet with four sections
(**projector**, **aperture**, **combiner**, **target pupil**). :data:`CONFIG`
is the single instance this example builds from, and its
:meth:`~SystemConfig.describe` readout is printed to the console *and*
overlaid on the rendered figure. Values the library fixes internally
(mirror tilt, prism skew, eye relief, aperture frame, pupil placement)
are imported from :mod:`helios.combiner_params` rather than re-typed, so
the sheet cannot drift; :meth:`~SystemConfig.verify_against_system`
cross-checks it against the system actually built.

Run::

    python examples/visualize_flat_system.py
"""

from dataclasses import dataclass

import math
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from apollo14.combiner import (
    DEFAULT_LIGHT_POSITION, DEFAULT_LIGHT_DIRECTION,
    compensated_reflectances,
)
from apollo14.elements.aperture import RectangularAperture
from apollo14.elements.partial_mirror import PartialMirror
from apollo14.elements.pupil import RectangularPupil
from apollo14.projector import FovGrid, PlayNitrideLed
from apollo14.route import combiner_main_path
from apollo14.spectral import ConstantCurve
from apollo14.trace import prepare_route, trace_rays
from apollo14.units import mm, nm, deg
from apollo14.visualizer import eyebox_overlay_traces, plot_system

from helios.combiner_params import (
    CombinerParams, build_parametrized_system,
    CHASSIS_X, CHASSIS_Y, SKEW_ANGLE, MIRROR_ANGLE, EYE_RELIEF,
    FIRST_MIRROR_OFFSET_Y, PUPIL_OFFSET_X, PUPIL_OFFSET_Y,
)
from helios.merit import build_combiner_branch_routes

# One JIT compile per distinct route structure (main + 8 branches),
# reused across every scan angle.
_trace_rays_jit = jax.jit(trace_rays)


# ── Formatting helpers ───────────────────────────────────────────────────────

def _len(value) -> str:
    return f"{value / mm:.2f} mm"


def _angle(value) -> str:
    return f"{value / deg:.1f} deg"


def _xyz(vec) -> str:
    vec = np.asarray(vec)
    return f"[{vec[0] / mm:.2f}, {vec[1] / mm:.2f}, {vec[2] / mm:.2f}] mm"


def _unit(vec) -> str:
    vec = np.asarray(vec)
    return f"[{vec[0]:+.2f}, {vec[1]:+.2f}, {vec[2]:+.2f}]"


def _section(title: str, rows: list[tuple[str, str]]) -> str:
    label_width = max(len(label) for label, _ in rows)
    lines = [f"── {title} " + "─" * max(0, 38 - len(title))]
    lines += [f"  {label:<{label_width}} : {value}" for label, value in rows]
    return "\n".join(lines)


# ── System configuration — the spec sheet ────────────────────────────────────
# All settings, grouped into the four sections the visualization draws.
# Fields the flat driver actively chooses are plain values here; fields the
# library builder fixes are imported from helios.combiner_params (above) so
# this sheet stays in lock-step with what build_parametrized_system produces.


@dataclass(frozen=True)
class ProjectorSettings:
    """The projector: where it sits, the beam it emits, how it is scanned."""
    position: jnp.ndarray           # emitter location
    normal: jnp.ndarray             # emission direction (beam axis)
    beam_width: float               # beam cross-section, local x
    beam_height: float              # beam cross-section, local y
    rays_x: int                     # ray-grid columns (visualization density)
    rays_y: int                     # ray-grid rows
    falloff_x: float                # angular intensity roll-off, local x (/rad)
    falloff_y: float                # angular intensity roll-off, local y (/rad)
    fov_x: float                    # FOV scan range, x
    fov_y: float                    # FOV scan range, y
    scan_steps_x: int               # slider steps across fov_x
    scan_steps_y: int               # slider steps across fov_y
    spectrum: str                   # emission-spectrum label

    def rows(self, peak_nm: float) -> list[tuple[str, str]]:
        return [
            ("location", _xyz(self.position)),
            ("normal", _unit(self.normal)),
            ("beam size", f"{_len(self.beam_width)} x {_len(self.beam_height)}"),
            ("beam rays", f"{self.rays_x} x {self.rays_y}"),
            ("angular falloff",
             f"x={self.falloff_x * deg:.3f}/deg  y={self.falloff_y * deg:.3f}/deg"),
            ("projection angle range",
             f"±{self.fov_x / deg / 2:.1f} x ±{self.fov_y / deg / 2:.1f} deg"),
            ("scan steps", f"{self.scan_steps_x} x {self.scan_steps_y}"),
            ("spectrum", f"{self.spectrum} (peak {peak_nm:.0f} nm)"),
        ]


@dataclass(frozen=True)
class ApertureSettings:
    """The beam-defining stop: an opaque frame with a clear inner opening.

    Fixed inside ``build_parametrized_system`` — mirrored here so the spec
    sheet is complete; :meth:`SystemConfig.verify_against_system` asserts
    the two agree.
    """
    position: jnp.ndarray
    normal: jnp.ndarray
    outer_width: float
    outer_height: float
    inner_width: float              # clear opening = beam-defining aperture
    inner_height: float

    def rows(self) -> list[tuple[str, str]]:
        return [
            ("location", _xyz(self.position)),
            ("normal", _unit(self.normal)),
            ("outer frame",
             f"{_len(self.outer_width)} x {_len(self.outer_height)}"),
            ("aperture opening",
             f"{_len(self.inner_width)} x {_len(self.inner_height)}"),
        ]


@dataclass(frozen=True)
class CombinerSettings:
    """The glass chassis and its cascade of tilted partial mirrors."""
    chassis_size: tuple[float, float, float]   # (x, y, z)
    chassis_center: jnp.ndarray
    material: str
    mirror_angle: float             # mirror plane vs. horizontal (tilt)
    prism_angle: float              # chassis z-skew (wedge)
    num_mirrors: int
    distance_between_mirrors: float  # perpendicular gap between mirrors
    first_mirror_offset_y: float    # mirror_0 offset above chassis center
    base_ratio: float               # per-mirror pick-off fraction of the beam
    mirror_curves: str              # reflectance model

    @property
    def mirror_size(self) -> tuple[float, float]:
        """(width, height) of each mirror — width spans the chassis in x,
        height is the chassis thickness projected along the mirror tilt."""
        width = self.chassis_size[0]
        height = self.chassis_size[2] / math.cos(self.mirror_angle)
        return width, height

    def rows(self) -> list[tuple[str, str]]:
        size_x, size_y, size_z = self.chassis_size
        mirror_w, mirror_h = self.mirror_size
        return [
            ("size", f"{_len(size_x)} x {_len(size_y)} x {_len(size_z)}"),
            ("location (center)", _xyz(self.chassis_center)),
            ("material", self.material),
            ("prism angle", _angle(self.prism_angle)),
            ("num mirrors", str(self.num_mirrors)),
            ("distance between mirrors", _len(self.distance_between_mirrors)),
            ("mirror angle", _angle(self.mirror_angle)),
            ("mirror size", f"{_len(mirror_w)} x {_len(mirror_h)}"),
            ("first-mirror offset y", _len(self.first_mirror_offset_y)),
            ("per-mirror pick-off", f"{self.base_ratio:.4f}"),
            ("mirror curves", self.mirror_curves),
        ]


@dataclass(frozen=True)
class TargetPupilSettings:
    """The target-pupil plane and the eyebox region sampled on it."""
    center: jnp.ndarray             # pupil / eyebox center (world frame)
    normal: jnp.ndarray
    eye_relief: float               # pupil distance from the front face
    pupil_width: float              # target-pupil plane extent
    pupil_height: float
    half_x: float                   # eyebox half-extents (sampled region)
    half_y: float
    cells_x: int                    # eyebox sampling grid
    cells_y: int

    def rows(self) -> list[tuple[str, str]]:
        return [
            ("location", _xyz(self.center)),
            ("normal", _unit(self.normal)),
            ("eye relief", _len(self.eye_relief)),
            ("pupil size",
             f"{_len(self.pupil_width)} x {_len(self.pupil_height)}"),
            ("eyebox size",
             f"{_len(2 * self.half_x)} x {_len(2 * self.half_y)}"),
            ("eyebox cells", f"{self.cells_x} x {self.cells_y}"),
        ]


@dataclass(frozen=True)
class SystemConfig:
    """Every setting of the visualized system, in four labelled sections."""
    projector: ProjectorSettings
    aperture: ApertureSettings
    combiner: CombinerSettings
    target_pupil: TargetPupilSettings

    def describe(self, peak_nm: float) -> str:
        return "\n".join([
            _section("PROJECTOR", self.projector.rows(peak_nm)),
            "",
            _section("APERTURE", self.aperture.rows()),
            "",
            _section("COMBINER", self.combiner.rows()),
            "",
            _section("TARGET PUPIL", self.target_pupil.rows()),
        ])

    def verify_against_system(self, system) -> None:
        """Assert the duplicated/derived fields match the built system.

        Guards the two places this sheet restates library-internal
        geometry — the aperture frame and the pupil placement — plus the
        mirror count and size, so the printed spec can be trusted.
        """
        aperture = next(e for e in system.elements
                        if isinstance(e, RectangularAperture))
        pupil = next(e for e in system.elements
                     if isinstance(e, RectangularPupil))
        mirrors = [e for e in system.elements if isinstance(e, PartialMirror)]
        mirror_w, mirror_h = self.combiner.mirror_size

        scalar_checks = {
            "num mirrors": (len(mirrors), self.combiner.num_mirrors),
            "aperture outer_width": (float(aperture.width),
                                     self.aperture.outer_width),
            "aperture inner_width": (float(aperture.inner_width),
                                     self.aperture.inner_width),
            "pupil width": (float(pupil.width), self.target_pupil.pupil_width),
            "mirror width": (float(mirrors[0].width), mirror_w),
            "mirror height": (float(mirrors[0].height), mirror_h),
        }
        for name, (built, declared) in scalar_checks.items():
            if not np.isclose(built, declared, atol=1e-4):
                raise AssertionError(
                    f"config {name} ({declared}) != built ({built})")
        if not np.allclose(np.asarray(pupil.position),
                           np.asarray(self.target_pupil.center), atol=1e-4):
            raise AssertionError(
                f"config target-pupil center {_xyz(self.target_pupil.center)} "
                f"!= built pupil {_xyz(pupil.position)}")


# ── The instance this example builds from ────────────────────────────────────
# Flat-driver knobs (mirror count, spacing, chassis thickness, pick-off) are
# set here; the rest is imported from the library so it always matches.

_NUM_MIRRORS = 8
_CHASSIS_Z = 1.7 * mm                       # thin combiner (library default 2.0)
_CHASSIS_CENTER = jnp.array([CHASSIS_X / 2, CHASSIS_Y, _CHASSIS_Z / 2])

CONFIG = SystemConfig(
    projector=ProjectorSettings(
        position=DEFAULT_LIGHT_POSITION,
        normal=DEFAULT_LIGHT_DIRECTION,
        beam_width=10.0 * mm,
        beam_height=2.0 * mm,
        rays_x=9, rays_y=3,                 # coarse grid → a legible picture
        falloff_x=0.0, falloff_y=0.0,
        fov_x=8.0 * deg, fov_y=8.0 * deg,
        scan_steps_x=5, scan_steps_y=5,     # coarser than the driver's 8x8
        spectrum="PlayNitride broadband white",
    ),
    aperture=ApertureSettings(             # must match build_parametrized_system
        position=DEFAULT_LIGHT_POSITION - jnp.array([0.0, 0.5 * mm, 0.0]),
        normal=DEFAULT_LIGHT_DIRECTION,
        outer_width=14.0 * mm, outer_height=6.0 * mm,
        inner_width=10.0 * mm, inner_height=2.0 * mm,
    ),
    combiner=CombinerSettings(
        chassis_size=(float(CHASSIS_X), float(CHASSIS_Y), float(_CHASSIS_Z)),
        chassis_center=_CHASSIS_CENTER,
        material="agc_m074",
        mirror_angle=MIRROR_ANGLE,
        prism_angle=SKEW_ANGLE,
        num_mirrors=_NUM_MIRRORS,
        distance_between_mirrors=0.93 * mm,
        first_mirror_offset_y=FIRST_MIRROR_OFFSET_Y,
        base_ratio=0.05 * 6 / _NUM_MIRRORS,
        mirror_curves="flat (ConstantCurve), Talos-compensated",
    ),
    target_pupil=TargetPupilSettings(
        center=jnp.array([CHASSIS_X / 2 + PUPIL_OFFSET_X,
                          CHASSIS_Y + PUPIL_OFFSET_Y,
                          EYE_RELIEF + _CHASSIS_Z]),
        normal=jnp.array([0.0, 0.0, -1.0]),
        eye_relief=EYE_RELIEF,
        pupil_width=14.0 * mm, pupil_height=18.0 * mm,
        half_x=4.0 * mm, half_y=5.0 * mm,   # 8 x 10 mm eyebox
        cells_x=8, cells_y=10,
    ),
)

REPORT_PATH = Path("examples/reports/visualize_flat_system.html")


# ── System / projector construction from the config ──────────────────────────

def build_projector(cfg: SystemConfig) -> PlayNitrideLed:
    proj = cfg.projector
    return PlayNitrideLed.create_broadband(
        position=proj.position, direction=proj.normal,
        beam_width=proj.beam_width, beam_height=proj.beam_height,
        nx=proj.rays_x, ny=proj.rays_y,
        falloff_x=proj.falloff_x, falloff_y=proj.falloff_y,
    )


def build_flat_system(cfg: SystemConfig, probe_wavelengths: jnp.ndarray):
    """Rebuild the flat driver's optical system at its warm-start design.

    Uses the same Talos-cascade compensation the optimizer warm-starts
    from — each mirror's flat reflectance is ``ratio / (1 - i*ratio)`` so
    equal absolute intensity reflects off every mirror — so the picture
    shows a physically sensible starting design, not a degenerate one.
    """
    comb = cfg.combiner
    compensated_amps = compensated_reflectances(
        comb.base_ratio, comb.num_mirrors, num_samples=1).reshape(-1)  # (M,)
    params = CombinerParams(
        spacings=jnp.full((comb.num_mirrors - 1,), comb.distance_between_mirrors),
        curves=ConstantCurve(amplitude=compensated_amps.astype(jnp.float32)),
    )
    return build_parametrized_system(
        params, probe_wavelengths=probe_wavelengths,
        chassis_z=comb.chassis_size[2])


def _spec_annotation(text: str) -> dict:
    """A monospace paper-space text panel carrying the spec sheet."""
    html = text.replace(" ", " ").replace("\n", "<br>")
    return dict(
        text=html, xref="paper", yref="paper", x=0.0, y=1.0,
        xanchor="left", yanchor="top", showarrow=False, align="left",
        font=dict(family="monospace", size=9, color="#e0e0e0"),
        bgcolor="rgba(20,20,30,0.75)", bordercolor="#888", borderwidth=1,
        borderpad=6,
    )


def main():
    projector = build_projector(CONFIG)

    # Representative trace wavelength = the projector's spectral peak. Flat
    # mirrors are wavelength-uniform, so one wavelength fully describes the
    # ray geometry (dispersion in the glass is a sub-pixel effect here).
    spec_wls, spec_rad = projector.spectrum
    viz_wavelength = float(spec_wls[int(jnp.argmax(spec_rad))])

    # Dense probe grid spanning the projector's emission band so the stored
    # mirror reflectance covers the trace wavelength.
    band_lo, band_hi = projector.spectral_band(threshold=0.10)
    probe_wavelengths = jnp.linspace(band_lo, band_hi, 100)

    system = build_flat_system(CONFIG, probe_wavelengths)
    CONFIG.verify_against_system(system)

    print(CONFIG.describe(peak_nm=viz_wavelength / nm))
    print()

    # Main transmitted path (through all mirrors, out the front) + one
    # reflected branch per mirror (reflect, out the top, onto the pupil),
    # resolved once at the representative wavelength.
    num_mirrors = CONFIG.combiner.num_mirrors
    main_route = prepare_route(combiner_main_path(system), viz_wavelength)
    branch_routes = [
        prepare_route(route, viz_wavelength)
        for route in build_combiner_branch_routes(system, num_mirrors=num_mirrors)
    ]
    routes = [main_route, *branch_routes]  # 1 + num_mirrors routes per angle

    fov_grid = FovGrid(
        CONFIG.projector.normal, CONFIG.projector.fov_x, CONFIG.projector.fov_y,
        num_x=CONFIG.projector.scan_steps_x, num_y=CONFIG.projector.scan_steps_y)

    # Trace every route at every scan direction, appending in angle-major
    # order (all routes for angle 0, then angle 1, ...) — the flat-list
    # layout plot_system's per-angle slider expects.
    trace_results = []
    for direction in fov_grid.flat_directions:
        beam = projector.generate_rays(
            direction=direction, wavelength=viz_wavelength)
        for route in routes:
            trace_results.append(
                _trace_rays_jit(route, beam))

    print(f"Traced {len(trace_results)} beams "
          f"({len(fov_grid)} angles x {len(routes)} routes). Rendering...")

    fig = plot_system(
        system,
        trace_results=trace_results,
        scan_angles=np.asarray(fov_grid.angles_grid),
        projector=projector,
        show=False,
    )

    # Eyebox wireframe on the target-pupil plane, centered on the pupil.
    for overlay in eyebox_overlay_traces(
            np.asarray(CONFIG.target_pupil.center),
            float(CONFIG.target_pupil.half_x), float(CONFIG.target_pupil.half_y),
            np.asarray(CONFIG.target_pupil.normal),
            box_label="eyebox (8x10 mm)",
            marker_label="eyebox center",
    ):
        fig.add_trace(overlay)

    # Overlay the full spec sheet as a panel on the figure itself.
    fig.add_annotation(_spec_annotation(CONFIG.describe(viz_wavelength / nm)))

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(REPORT_PATH))
    print(f"Saved: {REPORT_PATH}")
    fig.show()


if __name__ == "__main__":
    main()
