import jax.numpy as jnp

from apollo14.combiner import build_default_system, DEFAULT_LIGHT_POSITION, DEFAULT_LIGHT_DIRECTION, DEFAULT_WAVELENGTH
from apollo14.route import display_route
from apollo14.trace import trace_ray, trace_beam
from apollo14.elements.surface import PartialMirror
from apollo14.elements.glass_block import GlassBlock
from apollo14.elements.pupil import Pupil, RectangularPupil


def test_default_system_creates_elements():
    system = build_default_system()

    # 1 chassis + 1 aperture + 6 mirrors + 1 pupil = 9
    assert len(system.elements) == 9

    mirrors = [e for e in system.elements if isinstance(e, PartialMirror)]
    assert len(mirrors) == 6

    glass = [e for e in system.elements if isinstance(e, GlassBlock)]
    assert len(glass) == 1

    pupils = [e for e in system.elements if isinstance(e, (Pupil, RectangularPupil))]
    assert len(pupils) == 1


def test_trace_on_axis_ray():
    system = build_default_system()
    route = display_route(system, DEFAULT_WAVELENGTH)

    result = trace_ray(DEFAULT_LIGHT_POSITION, DEFAULT_LIGHT_DIRECTION, route)
    assert result.intensities.shape == (6,)
    assert jnp.any(result.valid), "On-axis ray should hit at least one mirror"
    assert float(result.total_intensity) > 0


def test_trace_beam():
    from apollo14.projector import Projector
    from apollo14.units import mm

    system = build_default_system()
    route = display_route(system, DEFAULT_WAVELENGTH)

    proj = Projector.uniform(
        DEFAULT_LIGHT_POSITION, DEFAULT_LIGHT_DIRECTION,
        4.0 * mm, 2.0 * mm, DEFAULT_WAVELENGTH, nx=3, ny=3)
    origins, _, _, _ = proj.generate_rays()

    result = trace_beam(origins, DEFAULT_LIGHT_DIRECTION, route, color_idx=0)
    assert result.intensities.shape == (9, 6)
    assert float(result.total_intensity.sum()) > 0
