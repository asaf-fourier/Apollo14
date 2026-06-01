"""Lock the eyebox per-cell brightness budget in the RGB optimizer driver."""


def test_per_cell_target_distributes_budget_over_active_cells():
    # Importing the example builds a small reference system at module load,
    # so do it lazily — it shouldn't slow collection of the rest of the suite.
    import examples.optimize_pupil_rgb as ex

    # The merit excludes the 4 corner cells, so the per-cell target must
    # divide the eyebox budget across the *active* cells. Dividing by the full
    # cell count makes every active cell aim low and the eyebox total
    # undershoot EYEBOX_TARGET (the bug this locks against).
    assert ex.NUM_ACTIVE_EYEBOX_CELLS == ex.NUM_EYEBOX_CELLS - 4
    achieved = ex.NUM_ACTIVE_EYEBOX_CELLS * ex.PER_CELL_TARGET
    assert abs(achieved - ex.EYEBOX_TARGET) < 1e-9
