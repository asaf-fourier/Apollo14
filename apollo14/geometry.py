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


def ray_rect_intersect(origin, direction, position, normal, local_x, local_y, half_extents):
    """Intersect a ray with a bounded rectangular plane. Pure JAX.

    Args:
        origin: (3,) ray start.
        direction: (3,) ray direction (normalized).
        position: (3,) center of the rectangle.
        normal: (3,) plane normal.
        local_x: (3,) first local axis on the plane.
        local_y: (3,) second local axis on the plane.
        half_extents: (2,) [half_width, half_height].

    Returns:
        hit: (3,) intersection point on the plane.
        t: scalar, ray parameter (distance along direction).
        in_bounds: bool, whether the hit is within the rectangle and t > 0.
    """
    denom = jnp.dot(direction, normal)
    t = jnp.dot(position - origin, normal) / (denom + 1e-30)
    hit = origin + jnp.maximum(t, 0.0) * direction

    delta = hit - position
    in_bounds = (
        (jnp.abs(jnp.dot(delta, local_x)) <= half_extents[0]) &
        (jnp.abs(jnp.dot(delta, local_y)) <= half_extents[1]) &
        (t > 0)
    )
    return hit, t, in_bounds


def ray_intersect_planar_seg(ray, seg):
    """Intersect ``ray`` with any seg exposing the standard planar fields.

    Wraps :func:`ray_rect_intersect` for the common case where every field
    comes straight off a planar segment (``position``, ``normal``,
    ``local_x``, ``local_y``, ``half_extents``). Duck-typed — works for
    ``FaceSeg``, ``_SingleMirror``, ``ReflectMirrorSeg``, and the outer
    bound of ``ApertureSeg``.
    """
    return ray_rect_intersect(
        ray.pos, ray.dir, seg.position, seg.normal,
        seg.local_x, seg.local_y, seg.half_extents,
    )


def compute_local_axes(normal):
    """Compute two orthogonal axes on the plane defined by normal."""
    unit_normal = normalize(normal)
    # Try global X, fall back to global Y if aligned with the plane normal.
    candidate = jnp.where(jnp.abs(unit_normal[0]) < 0.9,
                          jnp.array([1.0, 0.0, 0.0]),
                          jnp.array([0.0, 1.0, 0.0]))
    local_x = candidate - jnp.dot(candidate, unit_normal) * unit_normal
    local_x = normalize(local_x)
    local_y = jnp.cross(unit_normal, local_x)
    return local_x, local_y


def planar_grid_points(center, normal, half_x, half_y, nx, ny):
    """Generate a regular (nx, ny) grid of points on a plane.

    The grid spans ``[-half_x, half_x] × [-half_y, half_y]`` in the plane's
    local axes. Returns a flat ``(nx*ny, 3)`` array of world positions.
    """
    local_x, local_y = compute_local_axes(normal)
    xs = jnp.linspace(-half_x, half_x, nx)
    ys = jnp.linspace(-half_y, half_y, ny)
    grid_x, grid_y = jnp.meshgrid(xs, ys)  # (ny, nx)
    positions = (center[None, None, :]
                 + grid_x[:, :, None] * local_x[None, None, :]
                 + grid_y[:, :, None] * local_y[None, None, :])
    return positions.reshape(-1, 3)


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
