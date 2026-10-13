"""Tests for numeric scales shared by the portable Zemax HTML reports."""

import pytest

from apollo14.export.zemax.reporting.html_plotting import (
    format_compact_number,
    nice_nonnegative_scale,
    nice_scale,
)


@pytest.mark.parametrize(
    ("maximum", "expected_step", "expected_maximum"),
    [
        (0.9999, 0.25, 1.0),
        (2.5, 0.5, 2.5),
        (105.0, 25.0, 125.0),
        (0.000106, 0.000025, 0.000125),
    ],
)
def test_nonnegative_scales_use_rounded_steps(maximum, expected_step, expected_maximum):
    scale = nice_nonnegative_scale(maximum)

    assert scale["minimum"] == 0.0
    assert scale["step"] == pytest.approx(expected_step)
    assert scale["maximum"] == pytest.approx(expected_maximum)
    assert scale["ticks"][0] == 0.0
    assert scale["ticks"][-1] == pytest.approx(expected_maximum)
    assert all(
        right > left for left, right in zip(scale["ticks"], scale["ticks"][1:], strict=False)
    )


def test_scale_can_preserve_a_nonzero_range_and_format_clean_labels():
    scale = nice_scale(9.9, 10.1)

    assert scale == {
        "minimum": 9.9,
        "maximum": 10.1,
        "step": 0.05,
        "ticks": [9.9, 9.95, 10.0, 10.05, 10.1],
    }
    assert format_compact_number(0.9999) == "1"
    assert format_compact_number(0.25) == "0.25"
