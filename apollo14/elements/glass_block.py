"""Glass block element — refractive volume defined by planar faces."""

from dataclasses import dataclass, field
from typing import List, NamedTuple

import jax.numpy as jnp

from apollo14.geometry import (
    normalize, compute_local_axes, ray_rect_intersect, snell_refract,
)
from apollo14.materials import Material, air
from apollo14.ray import Ray


class FaceSeg(NamedTuple):
    """Refracting face (glass-block face or boundary plane).

    ``n1``/``n2`` hold ``MaterialData`` before ``prepare_route`` and scalar
    arrays after.
    """
    position: jnp.ndarray
    normal: jnp.ndarray
    local_x: jnp.ndarray
    local_y: jnp.ndarray
    half_extents: jnp.ndarray
    n1: jnp.ndarray
    n2: jnp.ndarray


@dataclass
class GlassFace:
    """One planar face of a glass block.

    Owns its geometry and a back-reference to the parent block's material
    (wired up by ``GlassBlock.__post_init__``). ``build_segment`` uses
    that back-reference to pick ``n1``/``n2`` depending on whether the
    ray is entering or leaving the glass.
    """
    name: str
    position: jnp.ndarray
    normal: jnp.ndarray
    vertices: jnp.ndarray

    def __post_init__(self):
        self.normal = normalize(self.normal)
        local_x, local_y = compute_local_axes(self.normal)
        self._local_x = local_x
        self._local_y = local_y
        deltas = self.vertices - self.position
        self._verts_x = jnp.array([jnp.dot(d, local_x) for d in deltas])
        self._verts_y = jnp.array([jnp.dot(d, local_y) for d in deltas])
        self._block_material: Material = None  # wired by GlassBlock

    @property
    def half_extents(self) -> jnp.ndarray:
        hw = jnp.max(jnp.abs(self._verts_x))
        hh = jnp.max(jnp.abs(self._verts_y))
        return jnp.stack([hw, hh])

    def build_segment(self, current_material, mode):
        glass = self._block_material
        if glass is None:
            raise RuntimeError(
                f"GlassFace '{self.name}' has no parent block material — "
                "construct it via GlassBlock so the back-reference is wired.")

        if current_material.name == glass.name:
            incoming, outgoing = glass, air
        else:
            incoming, outgoing = current_material, glass

        seg = FaceSeg(
            position=self.position,
            normal=self.normal,
            local_x=self._local_x,
            local_y=self._local_y,
            half_extents=self.half_extents,
            n1=incoming.data,
            n2=outgoing.data,
        )
        return seg, outgoing


def face_interact(seg: FaceSeg, ray: Ray, wavelength):
    """Refract through a glass face using Snell's law.

    ``seg.n1`` and ``seg.n2`` must be scalar arrays — callers use
    ``prepare_route`` to resolve them from ``MaterialData``.
    """
    alive_in = ray.intensity > 0

    hit, _, in_bounds = ray_rect_intersect(
        ray.pos, ray.dir, seg.position, seg.normal,
        seg.local_x, seg.local_y, seg.half_extents)

    facing = jnp.where(jnp.dot(ray.dir, seg.normal) < 0,
                       seg.normal, -seg.normal)
    refracted, is_tir = snell_refract(ray.dir, facing, seg.n1, seg.n2)

    valid = in_bounds & ~is_tir & alive_in
    out_intensity = jnp.where(valid, ray.intensity, 0.0)
    out_pos = jnp.where(valid, hit, ray.pos)
    out_dir = jnp.where(valid, refracted, ray.dir)
    return Ray(pos=out_pos, dir=out_dir, intensity=out_intensity), hit, valid


@dataclass
class GlassBlock:
    """A refractive glass volume defined by planar faces."""
    name: str
    position: jnp.ndarray
    material: Material
    faces: List[GlassFace] = field(default_factory=list)

    def __post_init__(self):
        for f in self.faces:
            f._block_material = self.material

    def get_face(self, name: str) -> GlassFace:
        for f in self.faces:
            if f.name == name:
                return f
        raise KeyError(f"No face named '{name}' in {self.name}. "
                       f"Available: {[f.name for f in self.faces]}")

    @classmethod
    def create_chassis(cls, name, x, y, z, material, z_skew=0.0):
        """Create an axis-aligned glass block (optionally skewed in z)."""
        hx, hy, hz = x / 2.0, y / 2.0, z / 2.0

        b_lf = jnp.array([-hx, -hy, -hz])
        b_rf = jnp.array([hx, -hy, -hz])
        b_rb = jnp.array([hx, hy, -hz])
        b_lb = jnp.array([-hx, hy, -hz])

        t_lf = jnp.array([-hx, -hy - z_skew, hz])
        t_rf = jnp.array([hx, -hy - z_skew, hz])
        t_rb = jnp.array([hx, hy - z_skew, hz])
        t_lb = jnp.array([-hx, hy - z_skew, hz])

        def _face(name, pos, normal, verts):
            return GlassFace(name=name, position=pos,
                             normal=jnp.array(normal, dtype=float),
                             vertices=jnp.stack(verts))

        def _face_from_edges(name, verts):
            pos = verts[0]
            e1 = normalize(verts[1] - verts[0])
            e2 = normalize(verts[3] - verts[0])
            n = normalize(jnp.cross(e1, e2))
            return GlassFace(name=name, position=pos, normal=n,
                             vertices=jnp.stack(verts))

        faces = [
            _face("bottom", b_lf, [0, 0, -1], [b_lf, b_rf, b_rb, b_lb]),
            _face("top", t_lf, [0, 0, 1], [t_lf, t_lb, t_rb, t_rf]),
            _face("left", b_lf, [-1, 0, 0], [b_lf, b_lb, t_lb, t_lf]),
            _face("right", b_rf, [1, 0, 0], [b_rb, b_rf, t_rf, t_rb]),
            _face_from_edges("front", [b_lf, b_rf, t_rf, t_lf]),
            _face_from_edges("back", [b_rb, b_lb, t_lb, t_rb]),
        ]

        return cls(name=name, position=jnp.array([0.0, 0.0, 0.0]),
                   material=material, faces=faces)

    def build_segment(self, current_material, mode):
        raise TypeError(
            "GlassBlock is a volume — resolve a named face with "
            "system.resolve(('<block>', '<face>')) for route building.")

    def translate(self, offset):
        """Return a new GlassBlock translated by offset."""
        new_faces = [
            GlassFace(
                name=f.name,
                position=f.position + offset,
                normal=f.normal,
                vertices=f.vertices + offset,
            )
            for f in self.faces
        ]
        return GlassBlock(
            name=self.name,
            position=self.position + offset,
            material=self.material,
            faces=new_faces,
        )
