import jax.numpy as jnp

from apollo14.combiner import CombinerConfig, build_system
from apollo14.tracer import trace_sequential, trace_mirrors_sequential
from apollo14.elements.surface import PartialMirror
from apollo14.elements.glass_block import GlassBlock
from apollo14.elements.pupil import Pupil


def test_default_config_creates_system():
    config = CombinerConfig.default()
    system = build_system(config)

    # Should have: 1 chassis + 1 aperture + 6 mirrors + 1 pupil = 9 elements
    assert len(system.elements) == 9

    mirrors = [e for e in system.elements if isinstance(e, PartialMirror)]
    assert len(mirrors) == 6

    glass = [e for e in system.elements if isinstance(e, GlassBlock)]
    assert len(glass) == 1

    pupils = [e for e in system.elements if isinstance(e, Pupil)]
    assert len(pupils) == 1


def test_trace_on_axis_ray():
    config = CombinerConfig.default()
    system = build_system(config)

    # Fire a ray from the projector position straight down
    origin = config.light.position
    direction = config.light.direction

    result = trace_sequential(system, origin, direction, config.light.wavelength)
    assert len(result.hits) > 0, "Ray should hit something"

    # Check that some hits are on mirrors
    mirror_hits = [h for h in result.hits if h.element_name.startswith("mirror_")]
    assert len(mirror_hits) > 0, "Ray should hit at least one mirror"


def test_sequential_trace():
    config = CombinerConfig.default()
    system = build_system(config)

    origin = config.light.position
    direction = config.light.direction

    result = trace_mirrors_sequential(system, origin, direction, config.light.wavelength)
    assert len(result.hits) > 0
