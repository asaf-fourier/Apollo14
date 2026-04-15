import jax.numpy as jnp

from apollo14.projector import Projector, scan_directions
from apollo14.geometry import normalize


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


def test_custom_intensity_map():
    imap = jnp.array([[0.0, 0.5], [1.0, 0.2]])  # (2, 2)
    proj = Projector(
        position=jnp.array([0.0, 10.0, 0.0]),
        direction=jnp.array([0.0, -1.0, 0.0]),
        beam_width=2.0, beam_height=2.0,
        intensity_map=imap,
    )
    ray = proj.generate_rays()
    assert ray.pos.shape == (4, 3)
    assert jnp.allclose(ray.intensity, imap.ravel())


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
