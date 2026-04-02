from dataclasses import dataclass, field
from typing import List, Optional

import jax.numpy as jnp
import numpy as np

from apollo14.interaction import Interaction
from apollo14.geometry import (
    normalize, ray_plane_intersection, compute_local_axes,
    point_in_polygon_2d, reflect, snell_refract,
)
from apollo14.materials import Material
from apollo14.units import EPSILON


@dataclass
class GlassFace:
    """One planar face of a glass block."""
    name: str
    position: jnp.ndarray   # (3,) point on the plane
    normal: jnp.ndarray     # (3,) outward-pointing normal
    vertices: jnp.ndarray   # (N, 3) ordered polygon vertices

    def __post_init__(self):
        self.normal = normalize(self.normal)
        local_x, local_y = compute_local_axes(self.normal)
        self._local_x = local_x
        self._local_y = local_y
        # Pre-compute 2D vertices
        deltas = self.vertices - self.position
        self._verts_x = jnp.array([jnp.dot(d, local_x) for d in deltas])
        self._verts_y = jnp.array([jnp.dot(d, local_y) for d in deltas])

    def is_point_on_face(self, point):
        delta = point - self.position
        if jnp.abs(jnp.dot(delta, self.normal)) > EPSILON:
            return False
        lx = jnp.dot(delta, self._local_x)
        ly = jnp.dot(delta, self._local_y)
        return point_in_polygon_2d(lx, ly, self._verts_x, self._verts_y)


@dataclass
class GlassBlock:
    """A refractive glass volume defined by planar faces."""
    name: str
    position: jnp.ndarray   # (3,) center
    material: Material
    faces: List[GlassFace] = field(default_factory=list)

    def is_point_inside(self, point):
        for face in self.faces:
            if jnp.dot(point - face.position, face.normal) > EPSILON:
                return False
        return True

    def find_intersection(self, origin, direction):
        """Returns (distance, hit_point, face_normal, face_name) or None."""
        origin_inside = self.is_point_inside(origin)
        best_dist = jnp.inf
        best_result = None

        for face in self.faces:
            t = ray_plane_intersection(origin, direction, face.normal, face.position)
            if t == jnp.inf or t <= EPSILON:
                continue

            hit = origin + t * direction
            if not face.is_point_on_face(hit):
                continue

            dot_dn = jnp.dot(direction, face.normal)
            # Inside → expect exiting (dot > 0), outside → expect entering (dot < 0)
            valid = (origin_inside and dot_dn > 0) or (not origin_inside and dot_dn < 0)
            if valid and t < best_dist:
                best_dist = t
                best_result = (t, hit, face.normal, face.name)

        return best_result

    def interact(self, origin, direction, intensity, wavelength, env_material):
        """Refract or TIR at the glass block surface.

        Returns list of (new_origin, new_direction, new_intensity, interaction_type).
        """
        result = self.find_intersection(origin, direction)
        if result is None:
            return []

        dist, hit, face_normal, face_name = result
        is_entering = jnp.dot(direction, face_normal) < 0

        if is_entering:
            n1 = float(env_material.n(wavelength))
            n2 = float(self.material.n(wavelength))
            normal_for_snell = face_normal
        else:
            n1 = float(self.material.n(wavelength))
            n2 = float(env_material.n(wavelength))
            normal_for_snell = -face_normal

        new_dir, is_tir = snell_refract(direction, normal_for_snell, n1, n2)
        interaction = Interaction.TIR if is_tir else (Interaction.ENTERING if is_entering else Interaction.EXITING)
        return [(hit, new_dir, intensity, interaction)]

    @classmethod
    def create_chassis(cls, name, x, y, z, material, z_skew=0.0):
        """Create an axis-aligned glass block (optionally skewed in z)."""
        hx, hy, hz = x / 2.0, y / 2.0, z / 2.0

        # Bottom vertices (z = -hz)
        b_lf = jnp.array([-hx, -hy, -hz])
        b_rf = jnp.array([hx, -hy, -hz])
        b_rb = jnp.array([hx, hy, -hz])
        b_lb = jnp.array([-hx, hy, -hz])

        # Top vertices (z = +hz, skewed in y)
        t_lf = jnp.array([-hx, -hy - z_skew, hz])
        t_rf = jnp.array([hx, -hy - z_skew, hz])
        t_rb = jnp.array([hx, hy - z_skew, hz])
        t_lb = jnp.array([-hx, hy - z_skew, hz])

        def _face(name, pos, normal, verts):
            return GlassFace(name=name, position=pos, normal=jnp.array(normal, dtype=float),
                             vertices=jnp.stack(verts))

        def _face_from_edges(name, verts):
            """Compute normal from edge cross product."""
            pos = verts[0]
            e1 = normalize(verts[1] - verts[0])
            e2 = normalize(verts[3] - verts[0])
            n = normalize(jnp.cross(e1, e2))
            return GlassFace(name=name, position=pos, normal=n, vertices=jnp.stack(verts))

        faces = [
            _face("bottom", b_lf, [0, 0, -1], [b_lf, b_rf, b_rb, b_lb]),
            _face("top", t_lf, [0, 0, 1], [t_lf, t_lb, t_rb, t_rf]),
            _face("left", b_lf, [-1, 0, 0], [b_lf, b_lb, t_lb, t_lf]),
            _face("right", b_rf, [1, 0, 0], [b_rb, b_rf, t_rf, t_rb]),
            _face_from_edges("front", [b_lf, b_rf, t_rf, t_lf]),
            _face_from_edges("back", [b_rb, b_lb, t_lb, t_rb]),
        ]

        center = jnp.array([0.0, 0.0, 0.0])
        return cls(name=name, position=center, material=material, faces=faces)

    def translate(self, offset):
        """Return a new GlassBlock translated by offset."""
        new_faces = []
        for f in self.faces:
            new_faces.append(GlassFace(
                name=f.name,
                position=f.position + offset,
                normal=f.normal,
                vertices=f.vertices + offset,
            ))
        return GlassBlock(
            name=self.name,
            position=self.position + offset,
            material=self.material,
            faces=new_faces,
        )
