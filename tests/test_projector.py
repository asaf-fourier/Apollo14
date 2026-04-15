import jax.numpy as jnp

from apollo14.projector import Projector, scan_directions
from apollo14.geometry import normalize
from apollo14.units import deg, nm


def test_uniform_projector_ray_count():
    proj = Projector.uniform(
        position=jnp.array([0.0, 0.0, 0.0]),
        direction=jnp.array([0.0, -1.0, 0.0]),
        beam_width=4.0, beam_height=2.0,
        nx=5, ny=3,
    )
    ray = proj.generate_rays()
    assert ray.pos.shape == (15, 3)
    assert ray.dir.shape == (3,)
    assert ray.intensity.shape == (15,)
    assert jnp.all(ray.intensity == 1.0)


def test_angular_falloff():
    # 2% drop at 6° along both axes.
    falloff = 0.02 / (6.0 * deg)
    proj = Projector.uniform(
        position=jnp.array([0.0, 10.0, 0.0]),
        direction=jnp.array([0.0, -1.0, 0.0]),
        beam_width=2.0, beam_height=2.0,
        nx=2, ny=2,
        falloff_x=falloff, falloff_y=falloff,
    )
    # Straight ahead → full intensity.
    on_axis = proj.generate_rays()
    assert jnp.allclose(on_axis.intensity, 1.0)
    # Tilted 6° around local_x (affects ay) → ~2% drop.
    tilted = jnp.array([jnp.sin(6.0 * deg), -jnp.cos(6.0 * deg), 0.0])
    off = proj.generate_rays(direction=tilted)
    assert jnp.allclose(off.intensity, 0.98, atol=1e-3)


# ── _angular_gain ──────────────────────────────────────────────────────────

def _proj(falloff_x=0.0, falloff_y=0.0):
    return Projector.uniform(
        position=jnp.array([0.0, 10.0, 0.0]),
        direction=jnp.array([0.0, -1.0, 0.0]),
        beam_width=2.0, beam_height=2.0,
        nx=2, ny=2,
        falloff_x=falloff_x, falloff_y=falloff_y,
    )


def test_angular_gain_returns_scalar():
    proj = _proj(falloff_x=0.1, falloff_y=0.1)
    gain = proj._angular_gain(jnp.array([0.0, -1.0, 0.0]))
    assert gain.shape == ()


def test_angular_gain_on_axis_is_one():
    # Zero falloff: always 1. Nonzero falloff: still 1 on-axis.
    for proj in (_proj(), _proj(falloff_x=0.5, falloff_y=0.5)):
        gain = proj._angular_gain(jnp.array([0.0, -1.0, 0.0]))
        assert jnp.isclose(gain, 1.0)


def test_angular_gain_axes_are_independent():
    # Tilt purely along x should only pick up falloff_x; same for y.
    fx = 0.02 / (6.0 * deg)
    fy = 0.05 / (6.0 * deg)
    proj = _proj(falloff_x=fx, falloff_y=fy)

    # Tilt 6° around base_y — moves direction into base_x.
    tilt_x = jnp.array([jnp.sin(6.0 * deg), -jnp.cos(6.0 * deg), 0.0])
    gain_x = proj._angular_gain(tilt_x)
    assert jnp.isclose(gain_x, 1.0 - 0.02, atol=1e-5)

    # Tilt 6° around base_x — moves direction into base_y.
    tilt_y = jnp.array([0.0, -jnp.cos(6.0 * deg), jnp.sin(6.0 * deg)])
    gain_y = proj._angular_gain(tilt_y)
    assert jnp.isclose(gain_y, 1.0 - 0.05, atol=1e-5)


def test_angular_gain_combined_axes():
    # Both axes tilted simultaneously → product of the two falloffs.
    f = 0.02 / (6.0 * deg)
    proj = _proj(falloff_x=f, falloff_y=f)
    d = normalize(jnp.array([jnp.sin(6.0 * deg),
                              -jnp.cos(6.0 * deg),
                              jnp.sin(6.0 * deg)]))
    # The small-angle projection gives ~6° on each axis → ~(0.98)^2.
    gain = proj._angular_gain(d)
    assert jnp.isclose(gain, 0.98 * 0.98, atol=2e-3)


def test_angular_gain_is_symmetric_in_sign():
    # Falloff uses |angle| — ±tilt must give identical gain.
    f = 0.1
    proj = _proj(falloff_x=f, falloff_y=f)
    pos = jnp.array([jnp.sin(6.0 * deg), -jnp.cos(6.0 * deg), 0.0])
    neg = jnp.array([-jnp.sin(6.0 * deg), -jnp.cos(6.0 * deg), 0.0])
    assert jnp.isclose(proj._angular_gain(pos), proj._angular_gain(neg))


def test_angular_gain_clipped_to_zero():
    # Very steep falloff + large tilt → clipped to 0, never negative.
    proj = _proj(falloff_x=100.0, falloff_y=0.0)
    d = jnp.array([jnp.sin(45.0 * deg), -jnp.cos(45.0 * deg), 0.0])
    gain = proj._angular_gain(d)
    assert gain >= 0.0
    assert jnp.isclose(gain, 0.0)


# ── Spectrum scaling ───────────────────────────────────────────────────────

def test_generate_rays_with_spectrum():
    wls = jnp.array([400.0, 500.0, 600.0]) * nm
    rad = jnp.array([0.2, 1.0, 0.3])
    proj = Projector.uniform(
        position=jnp.array([0.0, 10.0, 0.0]),
        direction=jnp.array([0.0, -1.0, 0.0]),
        beam_width=2.0, beam_height=2.0,
        nx=2, ny=2,
        spectrum=(wls, rad),
    )
    ray = proj.generate_rays(wavelength=500.0 * nm)
    assert jnp.allclose(ray.intensity, 1.0)
    ray = proj.generate_rays(wavelength=400.0 * nm)
    assert jnp.allclose(ray.intensity, 0.2)
    # No wavelength supplied → spectrum is ignored.
    ray = proj.generate_rays()
    assert jnp.allclose(ray.intensity, 1.0)


def test_rays_are_collimated():
    proj = Projector.uniform(
        position=jnp.array([7.0, 31.0, 1.0]),
        direction=jnp.array([0.0, -1.0, 0.0]),
        beam_width=4.0, beam_height=2.0,
        nx=4, ny=4,
    )
    ray = proj.generate_rays()
    # Direction is a single shared (3,) vector for a collimated beam.
    assert ray.dir.shape == (3,)


def test_ray_origins_span_beam():
    proj = Projector.uniform(
        position=jnp.array([0.0, 0.0, 0.0]),
        direction=jnp.array([0.0, -1.0, 0.0]),
        beam_width=4.0, beam_height=2.0,
        nx=10, ny=10,
    )
    ray = proj.generate_rays()
    # Origins should span roughly beam_width x beam_height
    span = ray.pos.max(axis=0) - ray.pos.min(axis=0)
    # The beam is in the plane perpendicular to direction (0,-1,0) → x-z plane
    assert span.max() > 3.5  # close to beam_width=4


def test_scan_directions_shape():
    base_dir = jnp.array([0.0, -1.0, 0.0])
    dirs, angles = scan_directions(base_dir, x_fov=0.1, y_fov=0.1, num_x=5, num_y=3)
    assert dirs.shape == (3, 5, 3)
    assert angles.shape == (3, 5, 2)


def test_scan_center_matches_base():
    base_dir = normalize(jnp.array([0.0, -1.0, 0.0]))
    dirs, _ = scan_directions(base_dir, x_fov=0.2, y_fov=0.2, num_x=3, num_y=3)
    center_dir = dirs[1, 1]  # middle of 3x3 grid
    assert jnp.allclose(center_dir, base_dir, atol=1e-5)
