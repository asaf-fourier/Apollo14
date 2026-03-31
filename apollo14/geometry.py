import jax.numpy as jnp

from apollo14.units import EPSILON, FP_EPSILON


def normalize(v):
    return v / jnp.linalg.norm(v)


def ray_plane_intersection(origin, direction, plane_normal, plane_point):
    """Returns distance from origin to plane hit, or inf if no hit."""
    denom = jnp.dot(direction, plane_normal)
    t = jnp.where(
        jnp.abs(denom) < EPSILON,
        jnp.inf,
        jnp.dot(plane_point - origin, plane_normal) / denom,
    )
    # Negative or zero means behind the ray
    return jnp.where(t > EPSILON, t, jnp.inf)


def reflect(direction, normal):
    """Reflect direction off surface with given normal."""
    return direction - 2.0 * jnp.dot(direction, normal) * normal


def snell_refract(direction, normal, n1, n2):
    """Snell's law refraction.

    Returns (refracted_direction, is_tir).
    normal must point into the medium the ray is entering (toward n2 side).
    If TIR, returns the reflected direction instead.
    """
    cos_i = -jnp.dot(normal, direction)
    ratio = n1 / n2
    sin_t2 = ratio ** 2 * (1.0 - cos_i ** 2)
    is_tir = sin_t2 > 1.0

    cos_t = jnp.sqrt(jnp.maximum(1.0 - sin_t2, 0.0))
    refracted = ratio * direction + (ratio * cos_i - cos_t) * normal
    refracted = normalize(refracted)

    reflected = reflect(direction, normal)

    out_dir = jnp.where(is_tir, reflected, refracted)
    return out_dir, is_tir


def compute_local_axes(normal):
    """Compute two orthogonal axes on the plane defined by normal."""
    n = normalize(normal)
    # Try global X, fall back to global Y if aligned
    candidate = jnp.where(jnp.abs(n[0]) < 0.9, jnp.array([1.0, 0.0, 0.0]), jnp.array([0.0, 1.0, 0.0]))
    local_x = candidate - jnp.dot(candidate, n) * n
    local_x = normalize(local_x)
    local_y = jnp.cross(n, local_x)
    return local_x, local_y


def point_in_rect(local_x_coord, local_y_coord, half_width, half_height):
    """Check if a point (in local 2D coords) is within a rectangle."""
    return (jnp.abs(local_x_coord) <= half_width + FP_EPSILON) & (
        jnp.abs(local_y_coord) <= half_height + FP_EPSILON
    )


def point_in_circle(local_x_coord, local_y_coord, radius):
    """Check if a point is within a circle centered at origin."""
    return local_x_coord ** 2 + local_y_coord ** 2 <= (radius + FP_EPSILON) ** 2


def point_in_polygon_2d(px, py, vertices_x, vertices_y):
    """Ray casting algorithm for point-in-polygon.

    vertices_x, vertices_y: (N,) arrays of polygon vertex coordinates.
    """
    n = vertices_x.shape[0]
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = vertices_x[i], vertices_y[i]
        xj, yj = vertices_x[j], vertices_y[j]
        crosses = ((yi <= py) & (yj > py)) | ((yj <= py) & (yi > py))
        if crosses:
            x_intersect = xi + (py - yi) / (yj - yi) * (xj - xi)
            if px < x_intersect:
                inside = not inside
        j = i
    return inside
