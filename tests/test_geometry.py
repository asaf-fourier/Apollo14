import jax.numpy as jnp
import pytest
from jax import grad, jit

from apollo14.elements.glass_block import GlassBlock, validate_reflectance_table
from apollo14.geometry import (
    compute_local_axes,
    normalize,
    point_in_circle,
    point_in_rect,
    ray_plane_intersection,
    ray_rect_intersect,
    reflect,
    snell_refract,
)
from apollo14.materials import agc_m074
from apollo14.spectral import SpectralTable


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


def test_compute_local_axes_y_aligns_with_world_up():
    """For a Z-facing surface (e.g. the eye pupil), local_y must point
    toward +world_Y so heatmap "up" matches world "up"."""
    _, ly = compute_local_axes(jnp.array([0.0, 0.0, -1.0]))
    assert ly[1] > 0.99
    _, ly = compute_local_axes(jnp.array([0.0, 0.0, 1.0]))
    assert ly[1] > 0.99
    # Tilted mirror — local_y should still have a positive Y component.
    _, ly = compute_local_axes(jnp.array([0.0, 0.669, 0.743]))
    assert ly[1] > 0.0


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


def test_snell_refract_tir_returns_reflected():
    # Past the critical angle the output direction is the reflected ray.
    direction = normalize(jnp.array([0.9, -0.1, 0.0]))
    normal = jnp.array([0.0, 1.0, 0.0])
    out_dir, is_tir = snell_refract(direction, normal, 1.5, 1.0)
    assert is_tir
    assert jnp.allclose(out_dir, reflect(direction, normal), atol=1e-6)


def test_snell_refract_tir_gradient_is_finite():
    # At TIR the refracted branch evaluates sqrt(0); a naive
    # ``sqrt(maximum(1 - sin_t2, 0))`` leaks a NaN gradient through the
    # downstream ``jnp.where`` even though the refracted value is discarded.
    normal = jnp.array([0.0, 0.0, -1.0])

    def out_sum(direction):
        out_dir, _ = snell_refract(direction, normal, 1.6, 1.0)
        return jnp.sum(out_dir)

    steep = normalize(jnp.array([jnp.sin(jnp.deg2rad(70.0)), 0.0,
                                 jnp.cos(jnp.deg2rad(70.0))]))
    _, is_tir = snell_refract(steep, normal, 1.6, 1.0)
    assert is_tir
    assert jnp.all(jnp.isfinite(grad(out_sum)(steep)))


def _rect():
    """A 4×4 rectangle on the z=5 plane, normal +Z, world-aligned axes."""
    return dict(
        position=jnp.array([0.0, 0.0, 5.0]),
        normal=jnp.array([0.0, 0.0, 1.0]),
        local_x=jnp.array([1.0, 0.0, 0.0]),
        local_y=jnp.array([0.0, 1.0, 0.0]),
        half_extents=jnp.array([2.0, 2.0]),
    )


def test_ray_rect_intersect_hits_within_bounds():
    hit, t, in_bounds = ray_rect_intersect(
        jnp.array([0.0, 0.0, 0.0]), jnp.array([0.0, 0.0, 1.0]), **_rect())
    assert bool(in_bounds)
    assert jnp.allclose(t, 5.0)
    assert jnp.allclose(hit, jnp.array([0.0, 0.0, 5.0]))


def test_ray_rect_intersect_outside_bounds():
    # Crosses the plane at x=10, well outside the ±2 rectangle.
    _, _, in_bounds = ray_rect_intersect(
        jnp.array([10.0, 0.0, 0.0]), jnp.array([0.0, 0.0, 1.0]), **_rect())
    assert not bool(in_bounds)


def test_ray_rect_intersect_near_parallel_returns_inf():
    # A ray parallel to the plane must report no intersection (t = inf), not
    # a spuriously large finite distance from dividing by a tiny epsilon.
    _, t, in_bounds = ray_rect_intersect(
        jnp.array([0.0, 0.0, 0.0]), jnp.array([1.0, 0.0, 0.0]), **_rect())
    assert not bool(in_bounds)
    assert jnp.isinf(t)


def test_chassis_faces_are_centered_on_their_vertex_polygons():
    """Face bounds must be measured from the polygon center, not a corner."""
    chassis = GlassBlock.create_chassis(
        name="chassis", x=14.0, y=20.0, z=2.0,
        material=agc_m074, z_skew=0.25,
    )

    for face in chassis.faces:
        expected_center = jnp.mean(face.vertices, axis=0)
        assert jnp.allclose(face.position, expected_center, atol=1e-6)

        deltas = face.vertices - expected_center
        expected_half_extents = jnp.array([
            jnp.max(jnp.abs(deltas @ face._local_x)),
            jnp.max(jnp.abs(deltas @ face._local_y)),
        ])
        assert jnp.allclose(face.half_extents, expected_half_extents,
                            atol=1e-6)


def test_perseus_ar_coating_is_on_all_chassis_faces_and_survives_transforms():
    from apollo14.perseus import (
        PERSEUS_AR_REFLECTANCE,
        PERSEUS_NUM_MIRRORS,
        build_perseus_geometry,
        spacings_for_count,
    )

    chassis = build_perseus_geometry(
        spacings=spacings_for_count(PERSEUS_NUM_MIRRORS)).chassis

    assert {face.name for face in chassis.faces} == {
        "bottom", "top", "left", "right", "front", "back"}
    for face in chassis.faces:
        assert jnp.allclose(face.coating_reflectance.values,
                            PERSEUS_AR_REFLECTANCE.values)

    moved = chassis.translate(jnp.array([1.0, 2.0, 3.0]))
    for original, transformed in zip(chassis.faces, moved.faces, strict=True):
        assert jnp.array_equal(transformed.coating_reflectance.wavelengths,
                               original.coating_reflectance.wavelengths)
        assert jnp.array_equal(transformed.coating_reflectance.values,
                               original.coating_reflectance.values)


def test_chassis_coating_range_is_validated_outside_jit_but_is_jit_safe():
    with pytest.raises(ValueError, match=r"within \[0, 1\]"):
        validate_reflectance_table(SpectralTable.constant(
            1.1, jnp.array([400.0, 700.0])))

    @jit
    def build_and_read(reflectance):
        chassis = GlassBlock.create_chassis(
            name="coated", x=1.0, y=1.0, z=1.0, material=agc_m074,
            coating_reflectance=SpectralTable.constant(
                reflectance, jnp.array([400.0, 700.0])))
        return chassis.get_face("back").coating_reflectance.values

    assert jnp.allclose(build_and_read(0.005), 0.005)
