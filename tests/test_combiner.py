import jax.numpy as jnp

from apollo14.combiner import CombinerConfig, build_system
from apollo14.tracer import trace_nonsequential
from apollo14.elements.surface import PartialMirror
from apollo14.elements.glass_block import GlassBlock
from apollo14.elements.pupil import Pupil, RectangularPupil
from apollo14.elements.boundary import BoundaryPlane


def test_default_config_creates_system():
    config = CombinerConfig.default()
    system = build_system(config)

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
    config = CombinerConfig.default()
    system = build_system(config)

    # Fire a ray from the projector position straight down
    origin = config.light.position
    direction = config.light.direction

    result = trace_nonsequential(system, origin, direction, config.light.wavelength)
    assert len(result.flat_hits()) > 0, "Ray should hit something"

    # Check that some hits are on mirrors
    mirror_hits = [h for h in result.flat_hits() if h.element_name.startswith("mirror_")]
    assert len(mirror_hits) > 0, "Ray should hit at least one mirror"


def test_jax_trace():
    from apollo14.jax_tracer import trace_ray, params_from_config

    config = CombinerConfig.default()
    params = params_from_config(config)
    n_glass = float(config.chassis.material.n(config.light.wavelength))

    pts, ints, valid = trace_ray(
        config.light.position, config.light.direction, n_glass, params)
    assert ints.shape == (6,)
    assert jnp.any(valid)
