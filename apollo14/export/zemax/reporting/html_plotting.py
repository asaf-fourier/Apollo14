"""Dependency-free numeric-scale helpers for portable HTML reports.

The Zemax report writers are copied out of the Apollo14 package and run on the
OpticStudio computer. Keeping scale selection here gives every generated
report the same rounded ticks without depending on NumPy or a plotting
library.
"""

import math

_NICE_STEP_MULTIPLIERS = (1.0, 2.0, 2.5, 5.0, 10.0)
REPORT_LAYOUT_VERSION = "horizontal-scale-units-v1"


def nice_scale(minimum, maximum, *, target_intervals=4, include_zero=False):
    """Return rounded bounds and ticks covering ``minimum`` through ``maximum``.

    Tick spacing is selected from 1, 2, 2.5, 5, or 10 times a power of ten.
    This keeps labels readable while retaining enough divisions to judge plot
    magnitudes. The returned mapping is directly JSON serializable.
    """
    minimum = _finite_float(minimum, "scale minimum")
    maximum = _finite_float(maximum, "scale maximum")
    if maximum < minimum:
        raise ValueError("Scale maximum must be greater than or equal to minimum.")
    target_intervals = _positive_integer(target_intervals, "scale target_intervals")

    if include_zero:
        minimum = min(0.0, minimum)
        maximum = max(0.0, maximum)
    if minimum == maximum == 0.0:
        return {
            "minimum": 0.0,
            "maximum": 0.0,
            "step": 0.0,
            "ticks": [0.0],
        }
    if minimum == maximum:
        padding = abs(minimum) * 0.1 or 1.0
        minimum -= padding
        maximum += padding

    step = _nearest_nice_step((maximum - minimum) / target_intervals)
    tolerance = 1.0e-12 * max(abs(minimum), abs(maximum), step)
    scale_minimum = math.floor((minimum + tolerance) / step) * step
    scale_maximum = math.ceil((maximum - tolerance) / step) * step
    interval_count = round((scale_maximum - scale_minimum) / step)
    ticks = [_clean_float(scale_minimum + index * step) for index in range(interval_count + 1)]
    return {
        "minimum": _clean_float(scale_minimum),
        "maximum": _clean_float(scale_maximum),
        "step": _clean_float(step),
        "ticks": ticks,
    }


def nice_nonnegative_scale(maximum, *, target_intervals=4):
    """Return a zero-based rounded scale that contains ``maximum``."""
    maximum = _finite_float(maximum, "scale maximum")
    if maximum < 0.0:
        raise ValueError("A non-negative scale cannot contain a negative maximum.")
    return nice_scale(
        0.0,
        maximum,
        target_intervals=target_intervals,
        include_zero=True,
    )


def format_compact_number(value, significant_digits=3):
    """Format display values compactly while retaining machine data elsewhere."""
    value = _finite_float(value, "display value")
    significant_digits = _positive_integer(significant_digits, "significant_digits")
    return format(value, f".{significant_digits}g").replace("e+", "e")


def _nearest_nice_step(rough_step):
    exponent = 10.0 ** math.floor(math.log10(rough_step))
    candidates = [multiplier * exponent for multiplier in _NICE_STEP_MULTIPLIERS]
    return min(
        candidates,
        key=lambda candidate: abs(math.log(candidate / rough_step)),
    )


def _clean_float(value):
    """Remove insignificant binary noise from display-scale metadata."""
    cleaned = float(format(float(value), ".12g"))
    return 0.0 if cleaned == 0.0 else cleaned


def _finite_float(value, name):
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite number.") from error
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number.")
    return result


def _positive_integer(value, name):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer.")
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be a positive integer.") from error
    if result <= 0 or result != value:
        raise ValueError(f"{name} must be a positive integer.")
    return result
