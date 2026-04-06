import jax.numpy as jnp

from apollo14.combiner import build_default_system, DEFAULT_LIGHT_POSITION, DEFAULT_LIGHT_DIRECTION, DEFAULT_WAVELENGTH
from apollo14.tracer import trace_nonsequential
from apollo14.elements.surface import PartialMirror
from apollo14.elements.glass_block import GlassBlock
from apollo14.elements.pupil import Pupil, RectangularPupil
from apollo14.elements.boundary import BoundaryPlane


def test_default_system_creates_elements():
    system = build_default_system()

    # 1 chassis + 1 aperture + 6 mirrors + 1 pupil + 6 boundary planes = 15
    assert len(system.elements) == 15

    mirrors = [e for e in system.elements if isinstance(e, PartialMirror)]
    assert len(mirrors) == 6

    glass = [e for e in system.elements if isinstance(e, GlassBlock)]
    assert len(glass) == 1

    pupils = [e for e in system.elements if isinstance(e, (Pupil, RectangularPupil))]
    assert len(pupils) == 1

    boundaries = [e for e in system.elements if isinstance(e, BoundaryPlane)]
    assert len(boundaries) == 6


def test_trace_on_axis_ray():
    system = build_default_system()

    result = trace_nonsequential(system, DEFAULT_LIGHT_POSITION, DEFAULT_LIGHT_DIRECTION,
                                  DEFAULT_WAVELENGTH)
    assert len(result.flat_hits()) > 0, "Ray should hit something"

    mirror_hits = [h for h in result.flat_hits() if h.element_name.startswith("mirror_")]
    assert len(mirror_hits) > 0, "Ray should hit at least one mirror"


def test_jax_trace():
    from apollo14.jax_tracer import trace_ray, params_from_system
    from apollo14.elements.glass_block import GlassBlock

    system = build_default_system()
    params = params_from_system(system, DEFAULT_WAVELENGTH)
    chassis = next(e for e in system.elements if isinstance(e, GlassBlock))
    n_glass = float(chassis.material.n(DEFAULT_WAVELENGTH))

    pts, ints, valid = trace_ray(
        DEFAULT_LIGHT_POSITION, DEFAULT_LIGHT_DIRECTION, n_glass, params)
    assert ints.shape == (6,)
    assert jnp.any(valid)
