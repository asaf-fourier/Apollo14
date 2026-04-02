import jax.numpy as jnp
import pytest

from apollo14.system import OpticalSystem
from apollo14.materials import air, agc_m074
from apollo14.elements.surface import PartialMirror
from apollo14.elements.glass_block import GlassBlock
from apollo14.elements.aperture import RectangularAperture
from apollo14.elements.pupil import Pupil
from apollo14.interaction import Interaction
from apollo14.tracer import trace_nonsequential
from apollo14.combiner import CombinerConfig, build_system
from apollo14.units import mm, nm


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def simple_mirror_system():
    """A single mirror at y=0 facing up, with a pupil above it."""
    system = OpticalSystem(env_material=air)
    system.add(PartialMirror(
        name="mirror",
        position=jnp.array([0.0, 0.0, 0.0]),
        normal=jnp.array([0.0, 1.0, 0.0]),
        width=10.0, height=10.0,
        reflection_ratio=0.5, transmission_ratio=0.5,
    ))
    system.add(Pupil(
        name="pupil",
        position=jnp.array([0.0, 10.0, 0.0]),
        normal=jnp.array([0.0, -1.0, 0.0]),
        radius=5.0,
    ))
    return system


@pytest.fixture
def two_mirror_system():
    """Two mirrors in sequence with a pupil — mimics a simplified combiner."""
    system = OpticalSystem(env_material=air)
    system.add(PartialMirror(
        name="mirror_0",
        position=jnp.array([0.0, 0.0, 0.0]),
        normal=jnp.array([0.0, 0.0, 1.0]),
        width=20.0, height=20.0,
        reflection_ratio=0.3, transmission_ratio=0.7,
    ))
    system.add(PartialMirror(
        name="mirror_1",
        position=jnp.array([0.0, 0.0, -5.0]),
        normal=jnp.array([0.0, 0.0, 1.0]),
        width=20.0, height=20.0,
        reflection_ratio=0.3, transmission_ratio=0.7,
    ))
    system.add(Pupil(
        name="pupil",
        position=jnp.array([0.0, 0.0, 10.0]),
        normal=jnp.array([0.0, 0.0, -1.0]),
        radius=5.0,
    ))
    return system


@pytest.fixture
def combiner_system():
    config = CombinerConfig.default()
    return config, build_system(config)


# ── trace_nonsequential tests ────────────────────────────────────────────────────

class TestTraceSequential:

    def test_ray_hits_mirror(self, simple_mirror_system):
        origin = jnp.array([0.0, -5.0, 0.0])
        direction = jnp.array([0.0, 1.0, 0.0])
        result = trace_nonsequential(simple_mirror_system, origin, direction, 550 * nm)
        mirror_hits = [h for h in result.flat_hits() if h.element_name == "mirror"]
        assert len(mirror_hits) > 0

    def test_mirror_splits_ray(self, simple_mirror_system):
        origin = jnp.array([0.0, -5.0, 0.0])
        direction = jnp.array([0.0, 1.0, 0.0])
        result = trace_nonsequential(simple_mirror_system, origin, direction, 550 * nm)
        interactions = {h.interaction for h in result.flat_hits() if h.element_name == "mirror"}
        assert Interaction.REFLECTED in interactions
        assert Interaction.TRANSMITTED in interactions

    def test_intensity_conservation_at_mirror(self, simple_mirror_system):
        origin = jnp.array([0.0, -5.0, 0.0])
        direction = jnp.array([0.0, 1.0, 0.0])
        result = trace_nonsequential(simple_mirror_system, origin, direction, 550 * nm)
        mirror_hits = [h for h in result.flat_hits() if h.element_name == "mirror"]
        total = sum(float(h.intensity) for h in mirror_hits)
        assert abs(total - 1.0) < 1e-5

    def test_reflected_ray_reaches_pupil(self, simple_mirror_system):
        """Ray going up hits mirror, reflected ray goes back down — won't reach pupil above.
        But transmitted ray continues up and should reach the pupil."""
        origin = jnp.array([0.0, -5.0, 0.0])
        direction = jnp.array([0.0, 1.0, 0.0])
        result = trace_nonsequential(simple_mirror_system, origin, direction, 550 * nm)
        assert result.pupil_hit is not None
        assert result.pupil_hit.element_name == "pupil"
        assert float(result.pupil_hit.intensity) == pytest.approx(0.5, abs=1e-5)

    def test_ray_misses_everything(self, simple_mirror_system):
        origin = jnp.array([100.0, -5.0, 0.0])
        direction = jnp.array([0.0, 1.0, 0.0])
        result = trace_nonsequential(simple_mirror_system, origin, direction, 550 * nm)
        assert len(result.flat_hits()) == 0
        assert result.pupil_hit is None

    def test_max_depth_limits_tracing(self, two_mirror_system):
        origin = jnp.array([0.0, 0.0, 5.0])
        direction = jnp.array([0.0, 0.0, -1.0])
        result = trace_nonsequential(two_mirror_system, origin, direction, 550 * nm, max_depth=1)
        # With max_depth=1, should only process first interaction's children
        assert len(result.flat_hits()) <= 3

    def test_min_intensity_stops_tracing(self, simple_mirror_system):
        origin = jnp.array([0.0, -5.0, 0.0])
        direction = jnp.array([0.0, 1.0, 0.0])
        result = trace_nonsequential(simple_mirror_system, origin, direction, 550 * nm,
                                  intensity=1e-8)
        assert len(result.flat_hits()) == 0

    def test_two_mirrors_both_split(self, two_mirror_system):
        origin = jnp.array([0.0, 0.0, 5.0])
        direction = jnp.array([0.0, 0.0, -1.0])
        result = trace_nonsequential(two_mirror_system, origin, direction, 550 * nm)
        m0_hits = [h for h in result.flat_hits() if h.element_name == "mirror_0"]
        m1_hits = [h for h in result.flat_hits() if h.element_name == "mirror_1"]
        assert len(m0_hits) > 0
        assert len(m1_hits) > 0

    def test_aperture_blocks_outside_rays(self):
        system = OpticalSystem(env_material=air)
        system.add(RectangularAperture(
            name="aperture",
            position=jnp.array([0.0, 0.0, 0.0]),
            normal=jnp.array([0.0, 0.0, 1.0]),
            width=2.0, height=2.0,
        ))
        system.add(Pupil(
            name="pupil",
            position=jnp.array([0.0, 0.0, -5.0]),
            normal=jnp.array([0.0, 0.0, 1.0]),
            radius=5.0,
        ))
        # Ray outside the aperture opening → absorbed
        origin = jnp.array([5.0, 0.0, 5.0])
        direction = jnp.array([0.0, 0.0, -1.0])
        result = trace_nonsequential(system, origin, direction, 550 * nm)
        assert any(h.element_name == "aperture" and h.interaction == Interaction.ABSORBED
                   for h in result.flat_hits())
        assert result.pupil_hit is None

    def test_aperture_passes_inside_rays(self):
        system = OpticalSystem(env_material=air)
        system.add(RectangularAperture(
            name="aperture",
            position=jnp.array([0.0, 0.0, 0.0]),
            normal=jnp.array([0.0, 0.0, 1.0]),
            width=2.0, height=2.0,
        ))
        system.add(Pupil(
            name="pupil",
            position=jnp.array([0.0, 0.0, -5.0]),
            normal=jnp.array([0.0, 0.0, 1.0]),
            radius=5.0,
        ))
        # Ray through the opening → reaches pupil
        origin = jnp.array([0.0, 0.0, 5.0])
        direction = jnp.array([0.0, 0.0, -1.0])
        result = trace_nonsequential(system, origin, direction, 550 * nm)
        assert result.pupil_hit is not None


# ── Combiner integration tests ────────────────────────────────────────────────

class TestCombinerTracing:

    def test_on_axis_hits_all_mirrors(self, combiner_system):
        config, system = combiner_system
        result = trace_nonsequential(
            system, config.light.position, config.light.direction,
            config.light.wavelength,
        )
        mirror_names = {h.element_name for h in result.flat_hits() if h.element_name.startswith("mirror_")}
        assert len(mirror_names) == 6

    def test_on_axis_reaches_pupil(self, combiner_system):
        config, system = combiner_system
        result = trace_nonsequential(
            system, config.light.position, config.light.direction,
            config.light.wavelength,
        )
        assert result.pupil_hit is not None

    def test_chassis_refraction_occurs(self, combiner_system):
        config, system = combiner_system
        result = trace_nonsequential(
            system, config.light.position, config.light.direction,
            config.light.wavelength,
        )
        chassis_hits = [h for h in result.flat_hits() if h.element_name == "chassis"]
        assert len(chassis_hits) > 0
        interactions = {h.interaction for h in chassis_hits}
        assert Interaction.ENTERING in interactions or Interaction.EXITING in interactions

    def test_hit_points_are_finite(self, combiner_system):
        config, system = combiner_system
        result = trace_nonsequential(
            system, config.light.position, config.light.direction,
            config.light.wavelength,
        )
        for hit in result.flat_hits():
            assert jnp.all(jnp.isfinite(hit.point))
            assert jnp.all(jnp.isfinite(hit.direction))
