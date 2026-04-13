"""Tests for SurfaceState, surface_step, and mirror_branch_origin."""

import jax
import jax.numpy as jnp
import pytest

from apollo14.surface import SurfaceState, surface_step, mirror_branch_origin
from apollo14.geometry import normalize


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_surface(*, position=None, normal=None, half_extents=None,
                  n1=1.0, n2=1.0, reflectance=None, kill_on_miss=True):
    """Build a SurfaceState with sane defaults for testing."""
    if position is None:
        position = jnp.array([0.0, 0.0, 0.0])
    if normal is None:
        normal = jnp.array([0.0, 0.0, 1.0])
    if half_extents is None:
        half_extents = jnp.array([5.0, 5.0])
    if reflectance is None:
        reflectance = jnp.zeros(3)
    return SurfaceState(
        position=jnp.asarray(position, dtype=jnp.float32),
        normal=normalize(jnp.asarray(normal, dtype=jnp.float32)),
        half_extents=jnp.asarray(half_extents, dtype=jnp.float32),
        local_x=jnp.array([1.0, 0.0, 0.0]),
        local_y=jnp.array([0.0, 1.0, 0.0]),
        n1=jnp.float32(n1),
        n2=jnp.float32(n2),
        reflectance=jnp.asarray(reflectance, dtype=jnp.float32),
        kill_on_miss=jnp.bool_(kill_on_miss),
    )


# ── SurfaceState construction ────────────────────────────────────────────────

class TestSurfaceStateConstruction:

    def test_field_count(self):
        s = _make_surface()
        assert len(s) == 9

    def test_field_shapes(self):
        s = _make_surface()
        assert s.position.shape == (3,)
        assert s.normal.shape == (3,)
        assert s.half_extents.shape == (2,)
        assert s.local_x.shape == (3,)
        assert s.local_y.shape == (3,)
        assert s.n1.shape == ()
        assert s.n2.shape == ()
        assert s.reflectance.shape == (3,)
        assert s.kill_on_miss.shape == ()

    def test_is_namedtuple(self):
        s = _make_surface()
        assert hasattr(s, '_fields')
        assert 'position' in s._fields

    def test_jax_pytree_roundtrip(self):
        s = _make_surface()
        leaves, treedef = jax.tree.flatten(s)
        s2 = jax.tree.unflatten(treedef, leaves)
        for f in SurfaceState._fields:
            assert jnp.array_equal(getattr(s, f), getattr(s2, f))

    def test_replace(self):
        s = _make_surface()
        s2 = s._replace(n1=jnp.float32(1.5))
        assert float(s2.n1) == 1.5
        assert float(s.n1) == 1.0


# ── surface_step as aperture (n1=n2=1, refl=0, kill_on_miss=True) ───────────

class TestApertureBehavior:

    def test_ray_through_opening(self):
        s = _make_surface(kill_on_miss=True)
        origin = jnp.array([0.0, 0.0, -10.0])
        direction = jnp.array([0.0, 0.0, 1.0])
        out_pos, out_dir, out_int, _, valid = \
            surface_step(s, origin, direction, 1.0, 0)
        assert valid
        assert jnp.allclose(out_int, 1.0)
        assert jnp.allclose(out_pos, jnp.array([0.0, 0.0, 0.0]), atol=1e-5)

    def test_ray_outside_opening_killed(self):
        s = _make_surface(half_extents=jnp.array([1.0, 1.0]), kill_on_miss=True)
        origin = jnp.array([10.0, 0.0, -10.0])
        direction = jnp.array([0.0, 0.0, 1.0])
        _, _, out_int, _, valid = \
            surface_step(s, origin, direction, 1.0, 0)
        assert not valid
        assert float(out_int) == 0.0

    def test_no_direction_change(self):
        s = _make_surface()
        origin = jnp.array([1.0, 1.0, -5.0])
        direction = normalize(jnp.array([0.1, 0.0, 1.0]))
        _, out_dir, _, _, valid = \
            surface_step(s, origin, direction, 1.0, 0)
        assert valid
        assert jnp.allclose(out_dir, direction, atol=1e-5)


# ── surface_step as refracting surface (n1≠n2, refl=0, kill=True) ───────────

class TestRefractBehavior:

    def test_normal_incidence_no_bend(self):
        s = _make_surface(n1=1.0, n2=1.5)
        origin = jnp.array([0.0, 0.0, -5.0])
        direction = jnp.array([0.0, 0.0, 1.0])
        _, out_dir, out_int, _, valid = \
            surface_step(s, origin, direction, 1.0, 0)
        assert valid
        assert jnp.allclose(out_dir, direction, atol=1e-5)
        assert jnp.allclose(out_int, 1.0)

    def test_oblique_incidence_bends(self):
        s = _make_surface(n1=1.0, n2=1.5)
        origin = jnp.array([-2.0, 0.0, -5.0])
        direction = normalize(jnp.array([0.3, 0.0, 1.0]))
        _, out_dir, _, _, valid = \
            surface_step(s, origin, direction, 1.0, 0)
        assert valid
        assert abs(float(out_dir[2])) > abs(float(direction[2]))

    def test_tir_invalidates(self):
        s = _make_surface(n1=1.5, n2=1.0, half_extents=jnp.array([50.0, 50.0]))
        origin = jnp.array([-20.0, 0.0, -1.0])
        direction = normalize(jnp.array([1.0, 0.0, 0.05]))
        _, _, _, _, valid = \
            surface_step(s, origin, direction, 1.0, 0)
        assert not valid

    def test_intensity_preserved(self):
        s = _make_surface(n1=1.0, n2=1.5)
        origin = jnp.array([0.0, 0.0, -5.0])
        direction = jnp.array([0.0, 0.0, 1.0])
        _, _, out_int, _, valid = \
            surface_step(s, origin, direction, 0.8, 0)
        assert valid
        assert jnp.allclose(out_int, 0.8)


# ── surface_step as mirror (n1=n2, refl>0, kill_on_miss=False) ──────────────

class TestMirrorBehavior:

    @pytest.fixture
    def mirror(self):
        return _make_surface(
            n1=1.5, n2=1.5,
            reflectance=jnp.array([0.05, 0.08, 0.03]),
            kill_on_miss=False,
        )

    def test_direction_unchanged(self, mirror):
        origin = jnp.array([0.0, 0.0, -5.0])
        direction = jnp.array([0.0, 0.0, 1.0])
        _, out_dir, _, _, valid = \
            surface_step(mirror, origin, direction, 1.0, 0)
        assert valid
        assert jnp.allclose(out_dir, direction, atol=1e-5)

    def test_transmitted_intensity_red(self, mirror):
        origin = jnp.array([0.0, 0.0, -5.0])
        direction = jnp.array([0.0, 0.0, 1.0])
        _, _, trans_int, _, valid = \
            surface_step(mirror, origin, direction, 1.0, color_idx=0)
        assert valid
        assert jnp.allclose(trans_int, 0.95)  # 1 - 0.05

    def test_transmitted_intensity_green(self, mirror):
        origin = jnp.array([0.0, 0.0, -5.0])
        direction = jnp.array([0.0, 0.0, 1.0])
        _, _, trans_int, _, valid = \
            surface_step(mirror, origin, direction, 1.0, color_idx=1)
        assert valid
        assert jnp.allclose(trans_int, 0.92)  # 1 - 0.08

    def test_transmitted_intensity_blue(self, mirror):
        origin = jnp.array([0.0, 0.0, -5.0])
        direction = jnp.array([0.0, 0.0, 1.0])
        _, _, trans_int, _, valid = \
            surface_step(mirror, origin, direction, 1.0, color_idx=2)
        assert valid
        assert jnp.allclose(trans_int, 0.97)  # 1 - 0.03

    def test_miss_keeps_intensity(self, mirror):
        origin = jnp.array([100.0, 0.0, -5.0])
        direction = jnp.array([0.0, 0.0, 1.0])
        _, _, out_int, _, valid = \
            surface_step(mirror, origin, direction, 0.6, 0)
        assert not valid
        assert jnp.allclose(out_int, 0.6)

    def test_miss_position_unchanged(self, mirror):
        origin = jnp.array([100.0, 0.0, -5.0])
        direction = jnp.array([0.0, 0.0, 1.0])
        out_pos, out_dir, _, _, valid = \
            surface_step(mirror, origin, direction, 1.0, 0)
        assert not valid
        assert jnp.allclose(out_pos, origin)
        assert jnp.allclose(out_dir, direction)

    def test_cascaded_transmission(self, mirror):
        origin = jnp.array([0.0, 0.0, -5.0])
        direction = jnp.array([0.0, 0.0, 1.0])

        # First mirror (r=0.05 for red)
        pos1, dir1, int1, _, _ = \
            surface_step(mirror, origin, direction, 1.0, 0)
        assert jnp.allclose(int1, 0.95)

        # Second mirror at z=5
        mirror2 = mirror._replace(position=jnp.array([0.0, 0.0, 5.0]))
        _, _, int2, _, _ = \
            surface_step(mirror2, pos1, dir1, int1, 0)
        assert jnp.allclose(int2, 0.9025, atol=1e-5)  # 0.95 * 0.95


# ── mirror_branch_origin ────────────────────────────────────────────────────

class TestMirrorBranchOrigin:

    @pytest.fixture
    def mirror(self):
        return _make_surface(
            n1=1.5, n2=1.5,
            reflectance=jnp.array([0.05, 0.08, 0.03]),
            kill_on_miss=False,
        )

    def test_reflected_direction(self, mirror):
        direction = jnp.array([0.0, 0.0, 1.0])
        hit = jnp.array([0.0, 0.0, 0.0])
        refl_dir, _ = mirror_branch_origin(mirror, direction, 1.0, hit, True, 0)
        assert jnp.allclose(refl_dir, jnp.array([0.0, 0.0, -1.0]), atol=1e-5)

    def test_reflected_intensity_red(self, mirror):
        direction = jnp.array([0.0, 0.0, 1.0])
        hit = jnp.array([0.0, 0.0, 0.0])
        _, refl_int = mirror_branch_origin(mirror, direction, 1.0, hit, True, 0)
        assert jnp.allclose(refl_int, 0.05)

    def test_reflected_intensity_green(self, mirror):
        direction = jnp.array([0.0, 0.0, 1.0])
        hit = jnp.array([0.0, 0.0, 0.0])
        _, refl_int = mirror_branch_origin(mirror, direction, 1.0, hit, True, 1)
        assert jnp.allclose(refl_int, 0.08)

    def test_reflected_intensity_blue(self, mirror):
        direction = jnp.array([0.0, 0.0, 1.0])
        hit = jnp.array([0.0, 0.0, 0.0])
        _, refl_int = mirror_branch_origin(mirror, direction, 1.0, hit, True, 2)
        assert jnp.allclose(refl_int, 0.03)

    def test_conservation(self, mirror):
        """Transmitted (from surface_step) + reflected = input."""
        origin = jnp.array([0.0, 0.0, -5.0])
        direction = jnp.array([0.0, 0.0, 1.0])
        for ci in range(3):
            _, _, trans, hit, valid = \
                surface_step(mirror, origin, direction, 0.7, ci)
            _, refl = mirror_branch_origin(mirror, direction, 0.7, hit, valid, ci)
            assert jnp.allclose(trans + refl, 0.7, atol=1e-6)

    def test_invalid_hit_zero_intensity(self, mirror):
        direction = jnp.array([0.0, 0.0, 1.0])
        hit = jnp.array([0.0, 0.0, 0.0])
        _, refl_int = mirror_branch_origin(mirror, direction, 1.0, hit, False, 0)
        assert float(refl_int) == 0.0

    def test_scales_with_input_intensity(self, mirror):
        direction = jnp.array([0.0, 0.0, 1.0])
        hit = jnp.array([0.0, 0.0, 0.0])
        _, refl_half = mirror_branch_origin(mirror, direction, 0.5, hit, True, 0)
        _, refl_full = mirror_branch_origin(mirror, direction, 1.0, hit, True, 0)
        assert jnp.allclose(refl_half, refl_full * 0.5, atol=1e-6)

    def test_oblique_reflection(self, mirror):
        direction = normalize(jnp.array([0.3, 0.0, 1.0]))
        hit = jnp.array([0.0, 0.0, 0.0])
        refl_dir, _ = mirror_branch_origin(mirror, direction, 1.0, hit, True, 0)
        # Reflected direction should have opposite z-component
        assert float(refl_dir[2]) < 0
        # x-component preserved (mirror normal is z)
        assert jnp.allclose(refl_dir[0], direction[0], atol=1e-5)


# ── surface_step as detector (n1=n2=1, refl=0, kill=True) ──────────────────

class TestDetectorBehavior:

    def test_hit_records_point(self):
        s = _make_surface(position=jnp.array([0.0, 0.0, 10.0]))
        origin = jnp.array([0.0, 0.0, 0.0])
        direction = jnp.array([0.0, 0.0, 1.0])
        out_pos, _, out_int, _, valid = \
            surface_step(s, origin, direction, 0.5, 0)
        assert valid
        assert jnp.allclose(out_pos, jnp.array([0.0, 0.0, 10.0]), atol=1e-5)
        assert jnp.allclose(out_int, 0.5)

    def test_miss_kills(self):
        s = _make_surface(
            position=jnp.array([0.0, 0.0, 10.0]),
            half_extents=jnp.array([1.0, 1.0]),
            kill_on_miss=True,
        )
        origin = jnp.array([50.0, 0.0, 0.0])
        direction = jnp.array([0.0, 0.0, 1.0])
        _, _, out_int, _, valid = \
            surface_step(s, origin, direction, 0.5, 0)
        assert not valid
        assert float(out_int) == 0.0


# ── Edge cases ───────────────────────────────────────────────────────────────

class TestEdgeCases:

    def test_zero_intensity_in(self):
        s = _make_surface(reflectance=jnp.array([0.1, 0.1, 0.1]))
        origin = jnp.array([0.0, 0.0, -5.0])
        direction = jnp.array([0.0, 0.0, 1.0])
        _, _, trans, hit, valid = \
            surface_step(s, origin, direction, 0.0, 0)
        assert valid
        assert float(trans) == 0.0

    def test_ray_from_behind(self):
        s = _make_surface(position=jnp.array([0.0, 0.0, 0.0]))
        origin = jnp.array([0.0, 0.0, 5.0])
        direction = jnp.array([0.0, 0.0, 1.0])
        _, _, _, _, valid = \
            surface_step(s, origin, direction, 1.0, 0)
        assert not valid


# ── Differentiability ────────────────────────────────────────────────────────

class TestDifferentiability:

    def test_grad_transmitted_wrt_reflectance(self):
        def loss(refl):
            s = _make_surface(n1=1.5, n2=1.5, reflectance=refl, kill_on_miss=False)
            origin = jnp.array([0.0, 0.0, -5.0])
            direction = jnp.array([0.0, 0.0, 1.0])
            _, _, trans, _, _ = surface_step(s, origin, direction, 1.0, 0)
            return trans

        grads = jax.grad(loss)(jnp.array([0.1, 0.1, 0.1]))
        # d(trans)/d(refl[0]) = -intensity = -1.0 for color_idx=0
        assert jnp.allclose(grads[0], -1.0, atol=1e-4)

    def test_grad_reflected_wrt_reflectance(self):
        def loss(refl):
            s = _make_surface(n1=1.5, n2=1.5, reflectance=refl, kill_on_miss=False)
            origin = jnp.array([0.0, 0.0, -5.0])
            direction = jnp.array([0.0, 0.0, 1.0])
            _, _, _, hit, valid = surface_step(s, origin, direction, 1.0, 0)
            _, refl_int = mirror_branch_origin(s, direction, 1.0, hit, valid, 0)
            return refl_int

        grads = jax.grad(loss)(jnp.array([0.1, 0.1, 0.1]))
        # d(refl_int)/d(refl[0]) = intensity = 1.0 for color_idx=0
        assert jnp.allclose(grads[0], 1.0, atol=1e-4)

    def test_grad_wrt_position(self):
        def loss(pos):
            s = _make_surface(position=pos)
            origin = jnp.array([0.0, 0.0, -5.0])
            direction = jnp.array([0.0, 0.0, 1.0])
            out_pos, _, _, _, _ = surface_step(s, origin, direction, 1.0, 0)
            return jnp.sum(out_pos)

        grads = jax.grad(loss)(jnp.array([0.0, 0.0, 0.0]))
        assert grads.shape == (3,)

    def test_grad_conservation(self):
        """d(trans + refl)/d(intensity) = 1.0"""
        def loss(intensity):
            s = _make_surface(
                reflectance=jnp.array([0.1, 0.1, 0.1]),
                n1=1.5, n2=1.5, kill_on_miss=False)
            origin = jnp.array([0.0, 0.0, -5.0])
            direction = jnp.array([0.0, 0.0, 1.0])
            _, _, trans, hit, valid = surface_step(s, origin, direction, intensity, 0)
            _, refl = mirror_branch_origin(s, direction, intensity, hit, valid, 0)
            return trans + refl

        grad = jax.grad(loss)(jnp.float32(1.0))
        assert jnp.allclose(grad, 1.0, atol=1e-4)
