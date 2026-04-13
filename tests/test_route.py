"""Tests for generic Route construction and display_route builder."""

import jax
import jax.numpy as jnp
import pytest

from apollo14.combiner import build_default_system
from apollo14.route import Route, display_route, _to_surface, _stack_surface_states
from apollo14.surface import SurfaceState
from apollo14.elements.surface import PartialMirror
from apollo14.elements.glass_block import GlassBlock
from apollo14.elements.pupil import RectangularPupil
from apollo14.elements.aperture import RectangularAperture
from apollo14.materials import air
from apollo14.units import nm


WAVELENGTH = 550.0 * nm


@pytest.fixture
def system():
    return build_default_system()


@pytest.fixture
def route(system):
    return display_route(system, WAVELENGTH)


# ── Route structure ──────────────────────────────────────────────────────────

class TestRouteIsNamedTuple:

    def test_fields(self):
        assert Route._fields == ('preamble', 'mirrors', 'branch')

    def test_is_pytree(self, route):
        leaves, treedef = jax.tree.flatten(route)
        route2 = jax.tree.unflatten(treedef, leaves)
        assert isinstance(route2, Route)

    def test_replace_mirrors(self, route):
        new_refl = route.mirrors.reflectance * 2
        new_mirrors = route.mirrors._replace(reflectance=new_refl)
        new_route = route._replace(mirrors=new_mirrors)
        assert jnp.allclose(new_route.mirrors.reflectance,
                            route.mirrors.reflectance * 2)


# ── Preamble ─────────────────────────────────────────────────────────────────

class TestPreamble:

    def test_shape(self, route):
        """Preamble has 2 elements: aperture + entry face."""
        assert route.preamble.position.shape == (2, 3)
        assert route.preamble.normal.shape == (2, 3)
        assert route.preamble.half_extents.shape == (2, 2)

    def test_all_fields_stacked(self, route):
        for f in SurfaceState._fields:
            arr = getattr(route.preamble, f)
            assert arr.shape[0] == 2, f"preamble.{f} not stacked"

    def test_reflectance_zero(self, route):
        """Preamble elements have no reflectance."""
        assert jnp.all(route.preamble.reflectance == 0)

    def test_entry_face_refracts(self, route, system):
        """Entry face has n1 < n2 (air → glass)."""
        chassis = next(e for e in system.elements if isinstance(e, GlassBlock))
        n_glass = chassis.material.n(WAVELENGTH)
        entry = route.preamble  # index 1 is entry face
        assert float(entry.n1[1]) < float(entry.n2[1])
        assert jnp.allclose(entry.n2[1], n_glass, atol=1e-3)

    def test_aperture_kill_on_miss(self, route, system):
        """If system has aperture, it should kill on miss."""
        apertures = [e for e in system.elements
                     if isinstance(e, RectangularAperture)]
        if apertures:
            assert route.preamble.kill_on_miss[0]  # aperture

    def test_entry_face_kill_on_miss(self, route):
        assert route.preamble.kill_on_miss[1]  # entry face


# ── Mirrors ──────────────────────────────────────────────────────────────────

class TestMirrors:

    def test_count(self, route, system):
        mirrors = [e for e in system.elements if isinstance(e, PartialMirror)]
        M = len(mirrors)
        assert route.mirrors.position.shape[0] == M

    def test_all_fields_stacked(self, route):
        M = route.mirrors.position.shape[0]
        for f in SurfaceState._fields:
            arr = getattr(route.mirrors, f)
            assert arr.shape[0] == M, f"mirrors.{f} not stacked to M={M}"

    def test_reflectance_nonzero(self, route):
        assert jnp.all(route.mirrors.reflectance > 0)

    def test_reflectance_per_color(self, route):
        assert route.mirrors.reflectance.shape[-1] == 3

    def test_n1_equals_n2(self, route):
        """Mirrors don't refract: n1 == n2 == n_glass."""
        assert jnp.allclose(route.mirrors.n1, route.mirrors.n2)

    def test_no_kill_on_miss(self, route):
        """Mirrors don't kill rays that miss — ray passes through."""
        assert not jnp.any(route.mirrors.kill_on_miss)

    def test_reflectance_matches_elements(self, route, system):
        """Route mirrors reflectance matches source element reflectance."""
        mirrors = [e for e in system.elements if isinstance(e, PartialMirror)]
        for i, m in enumerate(mirrors):
            assert jnp.allclose(route.mirrors.reflectance[i], m.state.reflectance)

    def test_positions_match_elements(self, route, system):
        mirrors = [e for e in system.elements if isinstance(e, PartialMirror)]
        for i, m in enumerate(mirrors):
            assert jnp.allclose(route.mirrors.position[i], m.state.position,
                                atol=1e-5)


# ── Branch ───────────────────────────────────────────────────────────────────

class TestBranch:

    def test_shape(self, route):
        """Branch has 2 elements: exit face + pupil."""
        assert route.branch.position.shape == (2, 3)

    def test_reflectance_zero(self, route):
        assert jnp.all(route.branch.reflectance == 0)

    def test_kill_on_miss(self, route):
        """Both branch elements kill on miss."""
        assert jnp.all(route.branch.kill_on_miss)

    def test_exit_face_refracts(self, route, system):
        """Exit face has n1 > n2 (glass → air)."""
        assert float(route.branch.n1[0]) > float(route.branch.n2[0])

    def test_pupil_no_refraction(self, route):
        """Pupil (index 1) has n1 == n2."""
        assert jnp.allclose(route.branch.n1[1], route.branch.n2[1])


# ── Helpers ──────────────────────────────────────────────────────────────────

class TestHelpers:

    def test_to_surface_defaults(self):
        s = _to_surface(
            jnp.zeros(3), jnp.array([0, 0, 1.]),
            jnp.array([5., 5.]), jnp.array([1., 0, 0]), jnp.array([0, 1., 0]))
        assert float(s.n1) == 1.0
        assert float(s.n2) == 1.0
        assert jnp.all(s.reflectance == 0)
        assert bool(s.kill_on_miss)

    def test_to_surface_custom(self):
        s = _to_surface(
            jnp.zeros(3), jnp.array([0, 0, 1.]),
            jnp.array([5., 5.]), jnp.array([1., 0, 0]), jnp.array([0, 1., 0]),
            n1=1.0, n2=1.5, reflectance=jnp.array([0.1, 0.2, 0.3]),
            kill_on_miss=False)
        assert float(s.n2) == 1.5
        assert jnp.allclose(s.reflectance, jnp.array([0.1, 0.2, 0.3]))
        assert not bool(s.kill_on_miss)

    def test_stack_surface_states(self):
        s1 = _to_surface(
            jnp.array([1., 0, 0]), jnp.array([0, 0, 1.]),
            jnp.array([5., 5.]), jnp.array([1., 0, 0]), jnp.array([0, 1., 0]))
        s2 = _to_surface(
            jnp.array([2., 0, 0]), jnp.array([0, 0, 1.]),
            jnp.array([5., 5.]), jnp.array([1., 0, 0]), jnp.array([0, 1., 0]))
        stacked = _stack_surface_states([s1, s2])
        assert stacked.position.shape == (2, 3)
        assert jnp.allclose(stacked.position[0], jnp.array([1., 0, 0]))
        assert jnp.allclose(stacked.position[1], jnp.array([2., 0, 0]))


# ── No-aperture system ───────────────────────────────────────────────────────

class TestNoApertureSystem:

    def test_dummy_aperture_does_not_block(self):
        """System without aperture: dummy aperture with huge extents, no kill."""
        system = build_default_system()
        # Remove apertures
        system.elements = [e for e in system.elements
                           if not isinstance(e, RectangularAperture)]
        route = display_route(system, WAVELENGTH)

        # Dummy aperture should have huge extents and kill_on_miss=False
        assert float(route.preamble.half_extents[0, 0]) > 1e4
        assert not bool(route.preamble.kill_on_miss[0])
