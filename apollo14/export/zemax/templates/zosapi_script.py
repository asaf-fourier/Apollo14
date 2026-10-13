"""Public accessors for the three generated Zemax Python scripts.

Each script template lives in a module named after the artifact it generates.
This façade keeps the API used by :mod:`apollo14.export.zemax.bundle` and
by downstream callers while making each generated program independently
reviewable and testable.
"""

from apollo14.export.zemax.templates.zosapi_build_template import BUILD_SCRIPT
from apollo14.export.zemax.templates.zosapi_runner_template import BUILD_AND_RUN_SCRIPT
from apollo14.export.zemax.templates.zosapi_sweep_template import SWEEP_SCRIPT


def build_script_text() -> str:
    """Return the ZOS-API model builder written into a bundle."""
    return BUILD_SCRIPT


def sweep_script_text() -> str:
    """Return the field-sweep and report-generation script."""
    return SWEEP_SCRIPT


def build_and_run_script_text() -> str:
    """Return the bundle wrapper that builds the model and runs the sweep."""
    return BUILD_AND_RUN_SCRIPT


__all__ = [
    "build_and_run_script_text",
    "build_script_text",
    "sweep_script_text",
]
