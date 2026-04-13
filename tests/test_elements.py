"""Unit tests for optical elements — State construction and jax_interact physics."""

import jax
import jax.numpy as jnp
import pytest

from apollo14.elements.surface import PartialMirror, MirrorState
from apollo14.elements.refracting_surface import RefractingSurface, RefractState
from apollo14.elements.pupil import RectangularPupil, DetectorState
from apollo14.elements.aperture import RectangularAperture, ApertureState
from apollo14.elements.glass_block import GlassBlock
from apollo14.materials import agc_m074
from apollo14.units import mm, nm


# ── Helpers ──────────────────────────────────────────────────────────────────

def _origin_above(z=10.0):
    return jnp.array([0.0, 0.0, z])

def _dir_down():
    return jnp.array([0.0, 0.0, -1.0])

def _dir_up():
    return jnp.array([0.0, 0.0, 1.0])


# ═══════════════════════════════════════════════════════════════════════════════
# PartialMirror
# ═══════════════════════════════════════════════════════════════════════════════

class TestPartialMirrorConstruction:

    def test_state_fields(self):
        m = PartialMirror("m0", position=jnp.array([0., 0., 0.]),
                          normal=jnp.array([0., 0., 1.]),
                          width=10.0, height=6.0,
                          reflectance=jnp.array([0.05, 0.08, 0.03]))
        s = m.state
        assert isinstance(s, MirrorState)
        assert s.position.shape == (3,)
        assert s.half_extents.shape == (2,)
        assert jnp.allclose(s.half_extents, jnp.array([5.0, 3.0]))
        assert s.reflectance.shape == (3,)

    def test_scalar_reflectance_broadcasts_to_3(self):
        m = PartialMirror("m0", position=jnp.zeros(3),
                          normal=jnp.array([0., 0., 1.]),
                          width=4.0, height=4.0,
                          reflectance=0.10)
        assert jnp.allclose(m.reflectance, jnp.array([0.10, 0.10, 0.10]))

    def test_default_reflectance(self):
        m = PartialMirror("m0", position=jnp.zeros(3),
                          normal=jnp.array([0., 0., 1.]),
                          width=4.0, height=4.0)
        assert jnp.allclose(m.reflectance, jnp.array([0.05, 0.05, 0.05]))

    def test_normal_is_normalized(self):
        m = PartialMirror("m0", position=jnp.zeros(3),
                          normal=jnp.array([0., 0., 5.]),
                          width=4.0, height=4.0)
        assert jnp.allclose(jnp.linalg.norm(m.normal), 1.0, atol=1e-6)

    def test_local_axes_orthogonal(self):
        m = PartialMirror("m0", position=jnp.zeros(3),
                          normal=jnp.array([1., 1., 1.]),
                          width=4.0, height=4.0)
        s = m.state
        assert jnp.abs(jnp.dot(s.local_x, s.local_y)) < 1e-5
        assert jnp.abs(jnp.dot(s.local_x, s.normal)) < 1e-5
        assert jnp.abs(jnp.dot(s.local_y, s.normal)) < 1e-5


class TestPartialMirrorInteract:

    @pytest.fixture
    def mirror_z0(self):
        """Mirror at z=0, normal pointing up, 10x10 mm."""
        m = PartialMirror("m0", position=jnp.zeros(3),
                          normal=jnp.array([0., 0., 1.]),
                          width=10.0, height=10.0,
                          reflectance=jnp.array([0.10, 0.20, 0.30]))
        return m.state

    def test_hit_in_bounds(self, mirror_z0):
        hit, _, trans_int, refl_dir, refl_int, valid = PartialMirror.jax_interact(
            mirror_z0, _origin_above(), _dir_down(), 1.0, color_idx=0)
        assert valid
        assert jnp.allclose(hit, jnp.zeros(3), atol=1e-5)

    def test_intensity_split_color_0(self, mirror_z0):
        """Color 0 (R) has reflectance 0.10."""
        _, _, trans_int, _, refl_int, _ = PartialMirror.jax_interact(
            mirror_z0, _origin_above(), _dir_down(), 1.0, color_idx=0)
        assert float(refl_int) == pytest.approx(0.10, abs=1e-5)
        assert float(trans_int) == pytest.approx(0.90, abs=1e-5)

    def test_intensity_split_color_2(self, mirror_z0):
        """Color 2 (B) has reflectance 0.30."""
        _, _, trans_int, _, refl_int, _ = PartialMirror.jax_interact(
            mirror_z0, _origin_above(), _dir_down(), 1.0, color_idx=2)
        assert float(refl_int) == pytest.approx(0.30, abs=1e-5)
        assert float(trans_int) == pytest.approx(0.70, abs=1e-5)

    def test_reflected_direction(self, mirror_z0):
        """Ray going down reflects back up off z-normal mirror."""
        _, _, _, refl_dir, _, _ = PartialMirror.jax_interact(
            mirror_z0, _origin_above(), _dir_down(), 1.0, color_idx=0)
        assert jnp.allclose(refl_dir, _dir_up(), atol=1e-5)

    def test_transmitted_direction_unchanged(self, mirror_z0):
        """Transmitted ray keeps the same direction."""
        _, trans_dir, _, _, _, _ = PartialMirror.jax_interact(
            mirror_z0, _origin_above(), _dir_down(), 1.0, color_idx=0)
        assert jnp.allclose(trans_dir, _dir_down(), atol=1e-5)

    def test_miss_outside_bounds(self, mirror_z0):
        """Ray hitting outside mirror bounds → not valid, no reflected intensity."""
        origin = jnp.array([100.0, 0.0, 10.0])  # far off to the side
        _, _, trans_int, _, refl_int, valid = PartialMirror.jax_interact(
            mirror_z0, origin, _dir_down(), 1.0, color_idx=0)
        assert not valid
        assert float(refl_int) == pytest.approx(0.0)
        assert float(trans_int) == pytest.approx(1.0)  # intensity preserved on miss

    def test_zero_intensity_in_zero_out(self, mirror_z0):
        _, _, trans_int, _, refl_int, _ = PartialMirror.jax_interact(
            mirror_z0, _origin_above(), _dir_down(), 0.0, color_idx=0)
        assert float(refl_int) == pytest.approx(0.0)
        assert float(trans_int) == pytest.approx(0.0)

    def test_differentiable_wrt_reflectance(self, mirror_z0):
        """Gradient of reflected intensity w.r.t. reflectance should be nonzero."""
        def refl_intensity(reflectance):
            s = mirror_z0._replace(reflectance=reflectance)
            _, _, _, _, refl_int, _ = PartialMirror.jax_interact(
                s, _origin_above(), _dir_down(), 1.0, color_idx=0)
            return refl_int

        grad = jax.grad(refl_intensity)(mirror_z0.reflectance)
        assert grad.shape == (3,)
        # d(refl_int)/d(reflectance[0]) = intensity = 1.0; others = 0
        assert float(grad[0]) == pytest.approx(1.0, abs=1e-4)
        assert float(grad[1]) == pytest.approx(0.0, abs=1e-4)


# ═══════════════════════════════════════════════════════════════════════════════
# RefractingSurface
# ═══════════════════════════════════════════════════════════════════════════════

class TestRefractingSurfaceConstruction:

    def test_state_fields(self):
        rs = RefractingSurface("entry", position=jnp.zeros(3),
                               normal=jnp.array([0., 0., 1.]),
                               half_width=5.0, half_height=3.0,
                               n1=1.0, n2=1.5)
        s = rs.state
        assert isinstance(s, RefractState)
        assert float(s.n1) == pytest.approx(1.0)
        assert float(s.n2) == pytest.approx(1.5)
        assert jnp.allclose(s.half_extents, jnp.array([5.0, 3.0]))


class TestRefractingSurfaceInteract:

    @pytest.fixture
    def glass_entry(self):
        """Air→glass interface at z=0, normal up."""
        rs = RefractingSurface("entry", position=jnp.zeros(3),
                               normal=jnp.array([0., 0., 1.]),
                               half_width=10.0, half_height=10.0,
                               n1=1.0, n2=1.5)
        return rs.state

    def test_normal_incidence_no_bend(self, glass_entry):
        """Ray hitting at normal incidence should pass straight through."""
        hit, new_dir, intensity, valid = RefractingSurface.jax_interact(
            glass_entry, _origin_above(), _dir_down(), 1.0)
        assert valid
        assert jnp.allclose(hit, jnp.zeros(3), atol=1e-5)
        # Normal incidence: direction unchanged (Snell: sin(0)=0 → sin(0)=0)
        assert jnp.allclose(new_dir, _dir_down(), atol=1e-4)

    def test_oblique_incidence_bends_toward_normal(self, glass_entry):
        """Entering denser medium at angle should bend toward normal."""
        from apollo14.geometry import normalize
        oblique = normalize(jnp.array([1.0, 0.0, -1.0]))  # 45 deg
        hit, new_dir, _, valid = RefractingSurface.jax_interact(
            glass_entry, jnp.array([-5.0, 0.0, 5.0]), oblique, 1.0)
        assert valid
        # In denser medium, angle from normal should decrease
        cos_in = jnp.abs(jnp.dot(oblique, jnp.array([0., 0., -1.])))
        cos_out = jnp.abs(jnp.dot(new_dir, jnp.array([0., 0., -1.])))
        assert float(cos_out) > float(cos_in)  # closer to normal = larger cosine

    def test_intensity_preserved(self, glass_entry):
        """Refraction doesn't change intensity (Fresnel losses not modeled)."""
        _, _, intensity, _ = RefractingSurface.jax_interact(
            glass_entry, _origin_above(), _dir_down(), 0.75)
        assert float(intensity) == pytest.approx(0.75)

    def test_miss_outside_bounds(self, glass_entry):
        origin = jnp.array([100.0, 0.0, 10.0])
        _, _, _, valid = RefractingSurface.jax_interact(
            glass_entry, origin, _dir_down(), 1.0)
        assert not valid

    def test_tir_detected(self):
        """Total internal reflection: glass→air at steep angle."""
        from apollo14.geometry import normalize
        rs = RefractingSurface("exit", position=jnp.zeros(3),
                               normal=jnp.array([0., 0., 1.]),
                               half_width=10.0, half_height=10.0,
                               n1=1.5, n2=1.0)
        # Nearly grazing angle (beyond critical angle for n=1.5→1.0)
        grazing = normalize(jnp.array([1.0, 0.0, -0.1]))
        _, _, _, valid = RefractingSurface.jax_interact(
            rs.state, jnp.array([0.0, 0.0, -5.0]), grazing, 1.0)
        assert not valid  # TIR → invalid


# ═══════════════════════════════════════════════════════════════════════════════
# RectangularPupil (Detector)
# ═══════════════════════════════════════════════════════════════════════════════

class TestRectangularPupilConstruction:

    def test_state_fields(self):
        p = RectangularPupil("pupil", position=jnp.array([0., 0., 15.]),
                             normal=jnp.array([0., 0., -1.]),
                             width=10.0, height=14.0)
        s = p.state
        assert isinstance(s, DetectorState)
        assert jnp.allclose(s.half_extents, jnp.array([5.0, 7.0]))

    def test_normal_is_normalized(self):
        p = RectangularPupil("pupil", position=jnp.zeros(3),
                             normal=jnp.array([0., 0., -3.]),
                             width=10.0, height=10.0)
        assert jnp.allclose(jnp.linalg.norm(p.normal), 1.0, atol=1e-6)


class TestRectangularPupilInteract:

    @pytest.fixture
    def pupil_z15(self):
        """Pupil at z=15, facing down, 10x14 mm."""
        p = RectangularPupil("pupil", position=jnp.array([0., 0., 15.]),
                             normal=jnp.array([0., 0., -1.]),
                             width=10.0, height=14.0)
        return p.state

    def test_hit_in_bounds(self, pupil_z15):
        hit, intensity, valid = RectangularPupil.jax_interact(
            pupil_z15, _origin_above(z=0.0), _dir_up(), 0.5)
        assert valid
        assert jnp.allclose(hit, jnp.array([0., 0., 15.]), atol=1e-4)
        assert float(intensity) == pytest.approx(0.5)

    def test_miss_outside_bounds(self, pupil_z15):
        origin = jnp.array([100.0, 0.0, 0.0])
        _, _, valid = RectangularPupil.jax_interact(
            pupil_z15, origin, _dir_up(), 1.0)
        assert not valid

    def test_miss_behind_ray(self, pupil_z15):
        """Pupil is at z=15, ray going down from z=20 → misses (going away)."""
        _, _, valid = RectangularPupil.jax_interact(
            pupil_z15, jnp.array([0., 0., 20.]), _dir_up(), 1.0)
        # Ray going up from z=20 hits z=15? No, z=15 is behind an upward ray from z=20.
        # Actually upward from z=20 goes to z>20, so it never hits z=15.
        # But the plane is at z=15 with normal (0,0,-1). dot(dir_up, normal) = -1 < 0
        # t = dot(pos - origin, normal) / denom = dot([0,0,-5], [0,0,-1]) / (-1) = 5/(-1) = -5
        # t < 0 → invalid
        assert not valid

    def test_intensity_unchanged(self, pupil_z15):
        """Detector doesn't modify intensity."""
        _, intensity, _ = RectangularPupil.jax_interact(
            pupil_z15, _origin_above(z=0.0), _dir_up(), 0.42)
        assert float(intensity) == pytest.approx(0.42)

    def test_width_height_bounds(self, pupil_z15):
        """Hit at edge of detector width (5mm) should be valid, just outside should not."""
        # At x=4.9 → valid
        _, _, valid_in = RectangularPupil.jax_interact(
            pupil_z15, jnp.array([4.9, 0., 0.]), _dir_up(), 1.0)
        assert valid_in
        # At x=5.1 → invalid
        _, _, valid_out = RectangularPupil.jax_interact(
            pupil_z15, jnp.array([5.1, 0., 0.]), _dir_up(), 1.0)
        assert not valid_out


# ═══════════════════════════════════════════════════════════════════════════════
# RectangularAperture
# ═══════════════════════════════════════════════════════════════════════════════

class TestRectangularApertureConstruction:

    def test_state_fields(self):
        a = RectangularAperture("ap", position=jnp.array([0., 30., 0.]),
                                normal=jnp.array([0., -1., 0.]),
                                width=4.0, height=1.0)
        s = a.state
        assert isinstance(s, ApertureState)
        assert jnp.allclose(s.half_extents, jnp.array([2.0, 0.5]))


class TestRectangularApertureClip:

    @pytest.fixture
    def aperture_y30(self):
        """Aperture at y=30, normal pointing -y, 4x1 mm opening."""
        a = RectangularAperture("ap", position=jnp.array([0., 30., 0.]),
                                normal=jnp.array([0., -1., 0.]),
                                width=4.0, height=1.0)
        return a.state

    def test_ray_through_opening(self, aperture_y30):
        """On-axis ray passes through the opening → intensity 1.0."""
        origin = jnp.array([0., 35., 0.])
        direction = jnp.array([0., -1., 0.])
        intensity = RectangularAperture.jax_clip(
            aperture_y30, origin, direction, has_aperture=jnp.array(True))
        assert float(intensity) == pytest.approx(1.0)

    def test_ray_blocked(self, aperture_y30):
        """Ray outside the aperture opening → intensity 0.0."""
        origin = jnp.array([10., 35., 0.])  # far off in x
        direction = jnp.array([0., -1., 0.])
        intensity = RectangularAperture.jax_clip(
            aperture_y30, origin, direction, has_aperture=jnp.array(True))
        assert float(intensity) == pytest.approx(0.0)

    def test_no_aperture_passes_everything(self, aperture_y30):
        """When has_aperture=False, even a blocked ray gets intensity 1.0."""
        origin = jnp.array([10., 35., 0.])  # would be blocked
        direction = jnp.array([0., -1., 0.])
        intensity = RectangularAperture.jax_clip(
            aperture_y30, origin, direction, has_aperture=jnp.array(False))
        assert float(intensity) == pytest.approx(1.0)

    def test_edge_of_opening(self, aperture_y30):
        """Ray at x=1.9 (inside 2.0 half-width) → passes."""
        origin = jnp.array([1.9, 35., 0.])
        direction = jnp.array([0., -1., 0.])
        intensity = RectangularAperture.jax_clip(
            aperture_y30, origin, direction, has_aperture=jnp.array(True))
        assert float(intensity) == pytest.approx(1.0)

    def test_just_outside_opening(self, aperture_y30):
        """Ray at x=2.1 (outside 2.0 half-width) → blocked."""
        origin = jnp.array([2.1, 35., 0.])
        direction = jnp.array([0., -1., 0.])
        intensity = RectangularAperture.jax_clip(
            aperture_y30, origin, direction, has_aperture=jnp.array(True))
        assert float(intensity) == pytest.approx(0.0)


# ═══════════════════════════════════════════════════════════════════════════════
# GlassBlock
# ═══════════════════════════════════════════════════════════════════════════════

class TestGlassBlock:

    @pytest.fixture
    def chassis(self):
        return GlassBlock.create_chassis(
            name="chassis", x=14.0, y=20.0, z=2.0,
            material=agc_m074, z_skew=0.0,
        ).translate(jnp.array([7.0, 10.0, 1.0]))

    def test_has_six_faces(self, chassis):
        assert len(chassis.faces) == 6

    def test_face_names(self, chassis):
        names = {f.name for f in chassis.faces}
        assert names == {"bottom", "top", "left", "right", "front", "back"}

    def test_get_face(self, chassis):
        f = chassis.get_face("back")
        assert f.name == "back"

    def test_get_face_missing(self, chassis):
        with pytest.raises(KeyError, match="no_such_face"):
            chassis.get_face("no_such_face")

    def test_face_produces_refracting_surface(self, chassis):
        rs = chassis.face("back", n1=1.0, n2=1.5)
        assert isinstance(rs, RefractingSurface)
        s = rs.state
        assert float(s.n1) == pytest.approx(1.0)
        assert float(s.n2) == pytest.approx(1.5)

    def test_face_half_extents_positive(self, chassis):
        for f in chassis.faces:
            hw, hh = f.half_extents
            assert hw > 0
            assert hh > 0

    def test_translate_moves_position(self):
        c1 = GlassBlock.create_chassis("c", x=10.0, y=10.0, z=2.0,
                                       material=agc_m074)
        c2 = c1.translate(jnp.array([5.0, 5.0, 5.0]))
        assert jnp.allclose(c2.position, jnp.array([5.0, 5.0, 5.0]))

    def test_translate_moves_faces(self):
        offset = jnp.array([3.0, 4.0, 5.0])
        c1 = GlassBlock.create_chassis("c", x=10.0, y=10.0, z=2.0,
                                       material=agc_m074)
        c2 = c1.translate(offset)
        for f1, f2 in zip(c1.faces, c2.faces):
            assert jnp.allclose(f2.position, f1.position + offset, atol=1e-5)

    def test_face_refracting_surface_interacts(self, chassis):
        """A ray aimed at the back face should hit and refract."""
        rs = chassis.face("back", n1=1.0, n2=1.5)
        # Back face is at large y; shoot a ray in -y direction from above
        origin = jnp.array([7.0, 25.0, 1.0])
        direction = jnp.array([0.0, -1.0, 0.0])
        hit, new_dir, _, valid = RefractingSurface.jax_interact(
            rs.state, origin, direction, 1.0)
        assert valid
