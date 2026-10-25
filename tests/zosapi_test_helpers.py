"""Shared test helpers for generated ZOS-API scripts."""

import types

from apollo14.export.zemax.templates.zosapi_script import build_script_text


def generated_builder_module():
    """Load the real generated builder into an in-memory test module."""
    module = types.ModuleType("build_zemax_model")
    module.__file__ = "/tmp/build_zemax_model.py"
    exec(build_script_text(), module.__dict__)
    return module
