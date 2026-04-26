import jax.numpy as jnp
from jax import grad

from apollo14.geometry import (
    compute_local_axes,
    normalize,
    point_in_circle,
    point_in_rect,
    ray_plane_intersection,
    reflect,
    snell_refract,
)


def test_normalize():
    v = jnp.array([3.0, 4.0, 0.0])
    n = normalize(v)
    assert jnp.allclose(jnp.linalg.norm(n), 1.0)
    assert jnp.allclose(n, jnp.array([0.6, 0.8, 0.0]))


def test_reflect():
    # 45-degree incidence on a horizontal surface
    direction = normalize(jnp.array([1.0, -1.0, 0.0]))
    normal = jnp.array([0.0, 1.0, 0.0])
    reflected = reflect(direction, normal)
    expected = normalize(jnp.array([1.0, 1.0, 0.0]))
    assert jnp.allclose(reflected, expected, atol=1e-6)


def test_snell_refract_normal_incidence():
    direction = jnp.array([0.0, -1.0, 0.0])
    normal = jnp.array([0.0, 1.0, 0.0])
    refracted, is_tir = snell_refract(direction, normal, 1.0, 1.5)
    assert not is_tir
    # Normal incidence → no bending
    assert jnp.allclose(refracted, direction, atol=1e-5)


def test_snell_refract_tir():
    # Glass to air at steep angle → TIR
    direction = normalize(jnp.array([0.9, -0.1, 0.0]))
    normal = jnp.array([0.0, 1.0, 0.0])
    _, is_tir = snell_refract(direction, normal, 1.5, 1.0)
    assert is_tir


def test_ray_plane_intersection():
    origin = jnp.array([0.0, 5.0, 0.0])
    direction = jnp.array([0.0, -1.0, 0.0])
    plane_normal = jnp.array([0.0, 1.0, 0.0])
    plane_point = jnp.array([0.0, 0.0, 0.0])
    dist = ray_plane_intersection(origin, direction, plane_normal, plane_point)
    assert jnp.allclose(dist, 5.0)


def test_ray_plane_parallel():
    origin = jnp.array([0.0, 5.0, 0.0])
    direction = jnp.array([1.0, 0.0, 0.0])  # parallel to plane
    plane_normal = jnp.array([0.0, 1.0, 0.0])
    plane_point = jnp.array([0.0, 0.0, 0.0])
    dist = ray_plane_intersection(origin, direction, plane_normal, plane_point)
    assert dist == jnp.inf


def test_compute_local_axes_orthogonal():
    normal = normalize(jnp.array([0.0, 1.0, 1.0]))
    lx, ly = compute_local_axes(normal)
    assert jnp.allclose(jnp.dot(lx, normal), 0.0, atol=1e-6)
    assert jnp.allclose(jnp.dot(ly, normal), 0.0, atol=1e-6)
    assert jnp.allclose(jnp.dot(lx, ly), 0.0, atol=1e-6)


def test_point_in_rect():
    assert point_in_rect(0.0, 0.0, 1.0, 1.0)
    assert not point_in_rect(1.5, 0.0, 1.0, 1.0)


def test_point_in_circle():
    assert point_in_circle(0.0, 0.0, 1.0)
    assert not point_in_circle(1.5, 0.0, 1.0)


def test_reflect_is_differentiable():
    def loss(angle):
        direction = jnp.array([jnp.sin(angle), -jnp.cos(angle), 0.0])
        normal = jnp.array([0.0, 1.0, 0.0])
        r = reflect(direction, normal)
        return r[0]  # x-component of reflection

    g = grad(loss)(jnp.float32(0.5))
    assert jnp.isfinite(g)
