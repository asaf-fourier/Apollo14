"""Lock the configuration of the flat-reflectance pupil optimizer driver."""


def test_eight_flat_mirrors_at_fixed_spacing():
    # Importing the example builds a small reference system at module load,
    # so do it lazily — it shouldn't slow collection of the rest of the suite.
    import examples.optimize_pupil_flat as ex
    from apollo14.elements.partial_mirror import PartialMirror

    # 8 mirrors, flat (wavelength-uniform) reflectance.
    assert ex.NUM_MIRRORS == 8
    mirrors = [e for e in ex._ref_system.elements
               if isinstance(e, PartialMirror)]
    assert len(mirrors) == 8

    # Every mirror's sampled reflectance is constant across the spectrum —
    # the defining property of the flat curve (no per-wavelength shape).
    for mirror in mirrors:
        reflectance = mirror.reflectance
        assert float(reflectance.max() - reflectance.min()) < 1e-6


def test_spacings_are_frozen_and_color_term_off():
    import examples.optimize_pupil_flat as ex

    # The geometry is fixed: spacings never participate in the update, so
    # the freeze must zero their gradient and the merit's color/shape term
    # must be disabled (flat mirrors can't change color balance).
    assert ex.merit_cfg.weight_shape == 0.0
    assert ex.merit_cfg.weight_target == 1.0

    import jax.numpy as jnp
    from helios.combiner_params import CombinerParams
    from apollo14.spectral import ConstantCurve
    from apollo14.units import mm

    params = CombinerParams(
        spacings=jnp.full((ex.NUM_MIRRORS - 1,), ex.FIXED_SPACING_MM * mm),
        curves=ConstantCurve.uniform(amplitude=0.05,
                                     num_mirrors=ex.NUM_MIRRORS),
    )
    frozen = ex._freeze_spacings(params)
    assert jnp.all(frozen.spacings == 0.0)


def test_per_cell_target_distributes_budget_over_active_cells():
    import examples.optimize_pupil_flat as ex

    # Same budget invariant as the RGB driver: the merit excludes the 4
    # corner cells, so the per-cell target must divide the eyebox budget
    # across the *active* cells or the eyebox total undershoots.
    assert ex.NUM_ACTIVE_EYEBOX_CELLS == ex.NUM_EYEBOX_CELLS - 4
    achieved = ex.NUM_ACTIVE_EYEBOX_CELLS * ex.PER_CELL_TARGET
    assert abs(achieved - ex.EYEBOX_TARGET) < 1e-9
