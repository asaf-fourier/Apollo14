import jax.numpy as jnp
import pytest

from apollo14.system import OpticalSystem
from apollo14.materials import air, agc_m074
from apollo14.elements.surface import PartialMirror
from apollo14.elements.glass_block import GlassBlock
from apollo14.elements.aperture import RectangularAperture
from apollo14.elements.pupil import Pupil
from apollo14.tracer import trace_sequential, trace_mirrors_sequential
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


# ── trace_sequential tests ────────────────────────────────────────────────────

class TestTraceSequential:

    def test_ray_hits_mirror(self, simple_mirror_system):
        origin = jnp.array([0.0, -5.0, 0.0])
        direction = jnp.array([0.0, 1.0, 0.0])
        result = trace_sequential(simple_mirror_system, origin, direction, 550 * nm)
        mirror_hits = [h for h in result.hits if h.element_name == "mirror"]
        assert len(mirror_hits) > 0

    def test_mirror_splits_ray(self, simple_mirror_system):
        origin = jnp.array([0.0, -5.0, 0.0])
        direction = jnp.array([0.0, 1.0, 0.0])
        result = trace_sequential(simple_mirror_system, origin, direction, 550 * nm)
        interactions = {h.interaction for h in result.hits if h.element_name == "mirror"}
        assert "reflected" in interactions
        assert "transmitted" in interactions

    def test_intensity_conservation_at_mirror(self, simple_mirror_system):
        origin = jnp.array([0.0, -5.0, 0.0])
        direction = jnp.array([0.0, 1.0, 0.0])
        result = trace_sequential(simple_mirror_system, origin, direction, 550 * nm)
        mirror_hits = [h for h in result.hits if h.element_name == "mirror"]
        total = sum(float(h.intensity) for h in mirror_hits)
        assert abs(total - 1.0) < 1e-5

    def test_reflected_ray_reaches_pupil(self, simple_mirror_system):
        """Ray going up hits mirror, reflected ray goes back down — won't reach pupil above.
        But transmitted ray continues up and should reach the pupil."""
        origin = jnp.array([0.0, -5.0, 0.0])
        direction = jnp.array([0.0, 1.0, 0.0])
        result = trace_sequential(simple_mirror_system, origin, direction, 550 * nm)
        assert result.pupil_hit is not None
        assert result.pupil_hit.element_name == "pupil"
        assert float(result.pupil_hit.intensity) == pytest.approx(0.5, abs=1e-5)

    def test_ray_misses_everything(self, simple_mirror_system):
        origin = jnp.array([100.0, -5.0, 0.0])
        direction = jnp.array([0.0, 1.0, 0.0])
        result = trace_sequential(simple_mirror_system, origin, direction, 550 * nm)
        assert len(result.hits) == 0
        assert result.pupil_hit is None

    def test_max_depth_limits_tracing(self, two_mirror_system):
        origin = jnp.array([0.0, 0.0, 5.0])
        direction = jnp.array([0.0, 0.0, -1.0])
        result = trace_sequential(two_mirror_system, origin, direction, 550 * nm, max_depth=1)
        # With max_depth=1, should only process first interaction's children
        assert len(result.hits) <= 3

    def test_min_intensity_stops_tracing(self, simple_mirror_system):
        origin = jnp.array([0.0, -5.0, 0.0])
        direction = jnp.array([0.0, 1.0, 0.0])
        result = trace_sequential(simple_mirror_system, origin, direction, 550 * nm,
                                  intensity=1e-8)
        assert len(result.hits) == 0

    def test_two_mirrors_both_split(self, two_mirror_system):
        origin = jnp.array([0.0, 0.0, 5.0])
        direction = jnp.array([0.0, 0.0, -1.0])
        result = trace_sequential(two_mirror_system, origin, direction, 550 * nm)
        m0_hits = [h for h in result.hits if h.element_name == "mirror_0"]
        m1_hits = [h for h in result.hits if h.element_name == "mirror_1"]
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
        result = trace_sequential(system, origin, direction, 550 * nm)
        assert any(h.element_name == "aperture" and h.interaction == "absorbed"
                   for h in result.hits)
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
        result = trace_sequential(system, origin, direction, 550 * nm)
        assert result.pupil_hit is not None


# ── trace_mirrors_sequential tests ────────────────────────────────────────────

class TestTraceMirrorsSequential:

    def test_follows_transmitted_path(self, two_mirror_system):
        origin = jnp.array([0.0, 0.0, 5.0])
        direction = jnp.array([0.0, 0.0, -1.0])
        result = trace_mirrors_sequential(two_mirror_system, origin, direction, 550 * nm)
        # Should record hits on both mirrors (transmitted path goes through both)
        m0 = [h for h in result.hits if h.element_name == "mirror_0"]
        m1 = [h for h in result.hits if h.element_name == "mirror_1"]
        assert len(m0) > 0
        assert len(m1) > 0

    def test_no_branching(self, two_mirror_system):
        """Sequential tracer should not branch — fewer hits than full DFS."""
        origin = jnp.array([0.0, 0.0, 5.0])
        direction = jnp.array([0.0, 0.0, -1.0])
        seq = trace_mirrors_sequential(two_mirror_system, origin, direction, 550 * nm)
        full = trace_sequential(two_mirror_system, origin, direction, 550 * nm)
        assert len(seq.hits) <= len(full.hits)

    def test_records_reflected_intensities(self, two_mirror_system):
        origin = jnp.array([0.0, 0.0, 5.0])
        direction = jnp.array([0.0, 0.0, -1.0])
        result = trace_mirrors_sequential(two_mirror_system, origin, direction, 550 * nm)
        reflected = [h for h in result.hits if h.interaction == "reflected"]
        assert len(reflected) >= 1
        for h in reflected:
            assert float(h.intensity) > 0

    def test_transmitted_intensity_decreases(self, two_mirror_system):
        origin = jnp.array([0.0, 0.0, 5.0])
        direction = jnp.array([0.0, 0.0, -1.0])
        result = trace_mirrors_sequential(two_mirror_system, origin, direction, 550 * nm)
        transmitted = [h for h in result.hits if h.interaction == "transmitted"]
        # Each transmission should have less intensity than the previous
        for i in range(1, len(transmitted)):
            assert float(transmitted[i].intensity) < float(transmitted[i - 1].intensity)


# ── Combiner integration tests ────────────────────────────────────────────────

class TestCombinerTracing:

    def test_on_axis_hits_all_mirrors(self, combiner_system):
        config, system = combiner_system
        result = trace_sequential(
            system, config.light.position, config.light.direction,
            config.light.wavelength,
        )
        mirror_names = {h.element_name for h in result.hits if h.element_name.startswith("mirror_")}
        assert len(mirror_names) == 6

    def test_on_axis_reaches_pupil(self, combiner_system):
        config, system = combiner_system
        result = trace_sequential(
            system, config.light.position, config.light.direction,
            config.light.wavelength,
        )
        assert result.pupil_hit is not None

    def test_sequential_records_all_mirrors(self, combiner_system):
        config, system = combiner_system
        result = trace_mirrors_sequential(
            system, config.light.position, config.light.direction,
            config.light.wavelength,
        )
        mirror_names = {h.element_name for h in result.hits if h.element_name.startswith("mirror_")}
        assert len(mirror_names) == 6

    def test_equal_reflection_per_mirror(self, combiner_system):
        """Each mirror should reflect the same global fraction (0.05)."""
        config, system = combiner_system
        result = trace_mirrors_sequential(
            system, config.light.position, config.light.direction,
            config.light.wavelength,
        )
        reflected = [h for h in result.hits
                     if h.element_name.startswith("mirror_") and h.interaction == "reflected"]
        for h in reflected:
            assert float(h.intensity) == pytest.approx(0.05, abs=1e-4)

    def test_total_reflected_intensity(self, combiner_system):
        """6 mirrors * 0.05 = 0.30 total reflected."""
        config, system = combiner_system
        result = trace_mirrors_sequential(
            system, config.light.position, config.light.direction,
            config.light.wavelength,
        )
        reflected = [h for h in result.hits
                     if h.element_name.startswith("mirror_") and h.interaction == "reflected"]
        total = sum(float(h.intensity) for h in reflected)
        assert total == pytest.approx(0.30, abs=1e-3)

    def test_remaining_transmitted_intensity(self, combiner_system):
        """After 6 mirrors at 0.05 each: 1.0 - 6*0.05 = 0.70 transmitted."""
        config, system = combiner_system
        result = trace_mirrors_sequential(
            system, config.light.position, config.light.direction,
            config.light.wavelength,
        )
        transmitted = [h for h in result.hits
                       if h.element_name.startswith("mirror_") and h.interaction == "transmitted"]
        last = transmitted[-1]
        assert float(last.intensity) == pytest.approx(0.70, abs=1e-3)

    def test_chassis_refraction_occurs(self, combiner_system):
        config, system = combiner_system
        result = trace_sequential(
            system, config.light.position, config.light.direction,
            config.light.wavelength,
        )
        chassis_hits = [h for h in result.hits if h.element_name == "chassis"]
        assert len(chassis_hits) > 0
        interactions = {h.interaction for h in chassis_hits}
        assert "entering" in interactions or "exiting" in interactions

    def test_hit_points_are_finite(self, combiner_system):
        config, system = combiner_system
        result = trace_sequential(
            system, config.light.position, config.light.direction,
            config.light.wavelength,
        )
        for hit in result.hits:
            assert jnp.all(jnp.isfinite(hit.point))
            assert jnp.all(jnp.isfinite(hit.direction))
