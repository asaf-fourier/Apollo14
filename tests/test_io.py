import json

import jax.numpy as jnp

from apollo14.projector import FovGrid, Projector
from apollo14.units import mm, nm
from helios.combiner_params import CombinerParams, build_parametrized_system
from helios.io import (
    _round_sig,
    _serialize_combiner_params,
    _serialize_element,
    _serialize_merit_config,
    _serialize_projector,
    _serialize_system,
    _to_list,
    save_optimization_report,
)
from helios.pupil_merit import PupilMeritConfig

# ── Helpers ────────────────────────────────────────────────────────────────


def _make_projector():
    return Projector.uniform(
        position=jnp.array([0.0, 10.0, 0.0]),
        direction=jnp.array([0.0, -1.0, 0.0]),
        beam_width=4.0 * mm, beam_height=2.0 * mm,
        nx=3, ny=3,
    )


# ── round_sig ─────────────────────────────────────────────────────────────


def test_round_sig_basic():
    assert _round_sig(1.23456789, 4) == 1.235


def test_round_sig_zero():
    assert _round_sig(0.0) == 0.0


def test_round_sig_small():
    assert _round_sig(0.000123456, 3) == 0.000123


# ── to_list ────────────────────────────────────────────────────────────────


def test_to_list_1d():
    result = _to_list(jnp.array([1.0, 2.0, 3.0]))
    assert isinstance(result, list)
    assert len(result) == 3


def test_to_list_2d():
    result = _to_list(jnp.array([[1.0, 2.0], [3.0, 4.0]]))
    assert len(result) == 2
    assert len(result[0]) == 2


# ── serialize_element ──────────────────────────────────────────────────────


def test_serialize_partial_mirror_with_curve():
    params = CombinerParams.initial()
    system = build_parametrized_system(params)
    mirrors = [e for e in system.elements
               if type(e).__name__ == "PartialMirror"]
    assert len(mirrors) == 6
    serialized = _serialize_element(mirrors[0])
    assert serialized["type"] == "PartialMirror"
    assert "position" in serialized
    assert "reflectance" in serialized
    # Curve parameters are nested under a discriminating sub-block.
    assert "curve" in serialized
    assert serialized["curve"]["type"] == "SumOfGaussiansCurve"
    assert len(serialized["curve"]["amplitude"]) == 3


def test_serialize_partial_mirror_base_fields():
    params = CombinerParams.initial()
    system = build_parametrized_system(params)
    mirror = next(e for e in system.elements
                  if type(e).__name__ == "PartialMirror")
    serialized = _serialize_element(mirror)
    assert "wavelengths" in serialized
    assert "width" in serialized
    assert "height" in serialized
    assert "normal" in serialized


def test_serialize_all_element_types():
    params = CombinerParams.initial()
    system = build_parametrized_system(params)
    for element in system.elements:
        serialized = _serialize_element(element)
        assert "type" in serialized
        assert "name" in serialized


# ── serialize_system ───────────────────────────────────────────────────────


def test_serialize_system_roundtrip_json():
    params = CombinerParams.initial()
    system = build_parametrized_system(params)
    serialized = _serialize_system(system)
    json_str = json.dumps(serialized)
    parsed = json.loads(json_str)
    assert len(parsed["elements"]) == len(system.elements)


# ── serialize_projector ────────────────────────────────────────────────────


def test_serialize_projector_fields():
    proj = _make_projector()
    serialized = _serialize_projector(proj)
    assert serialized["nx"] == 3
    assert serialized["ny"] == 3
    assert len(serialized["position"]) == 3


def test_serialize_projector_with_spectrum():
    wls = jnp.array([400.0, 500.0, 600.0]) * nm
    rad = jnp.array([0.2, 1.0, 0.3])
    proj = Projector.uniform(
        position=jnp.array([0.0, 10.0, 0.0]),
        direction=jnp.array([0.0, -1.0, 0.0]),
        beam_width=4.0, beam_height=2.0, nx=2, ny=2,
        spectrum=(wls, rad),
    )
    serialized = _serialize_projector(proj)
    assert serialized["spectrum"] is not None
    assert "wavelengths" in serialized["spectrum"]
    assert "radiance" in serialized["spectrum"]


# ── serialize_combiner_params ──────────────────────────────────────────────


def test_serialize_combiner_params():
    params = CombinerParams.initial()
    serialized = _serialize_combiner_params(params)
    assert "spacings" in serialized
    assert "curve" in serialized
    assert serialized["curve"]["type"] == "SumOfGaussiansCurve"
    assert len(serialized["spacings"]) == 5
    assert len(serialized["curve"]["amplitude"]) == 6


# ── serialize_merit_config ─────────────────────────────────────────────────


def test_serialize_merit_config():
    cfg = PupilMeritConfig(target_relative=0.05,
                           weight_target=2.0, weight_shape=0.5)
    serialized = _serialize_merit_config(cfg)
    assert serialized["target_relative"] == 0.05
    assert serialized["weight_target"] == 2.0
    assert serialized["weight_shape"] == 0.5


# ── save_optimization_report ───────────────────────────────────────────────


def test_save_optimization_report_creates_json(tmp_path):
    params = CombinerParams.initial()
    system = build_parametrized_system(params)
    proj = _make_projector()
    fov_grid = FovGrid(jnp.array([0.0, -1.0, 0.0]), 0.1, 0.1, num_x=3, num_y=3)

    from helios.combiner_params import ParamBounds
    report_path = save_optimization_report(
        tmp_path / "test_run",
        system=system,
        projectors=[proj],
        fov_grid=fov_grid,
        merit_config=PupilMeritConfig(),
        optimizer_config={"algorithm": "adam", "num_steps": 10},
        param_bounds=ParamBounds(),
        initial_params=params,
        final_params=params,
        initial_breakdown={"total": 1.0, "shape": 0.5},
        final_breakdown={"total": 0.5, "shape": 0.2},
        loss_history=[1.0, 0.9, 0.8],
        eyebox_config={"half_x": 5.0, "half_y": 5.0, "nx": 3, "ny": 3},
    )
    assert report_path.exists()
    data = json.loads(report_path.read_text())
    assert "git_sha" in data
    assert "timestamp" in data
    assert len(data["loss_history"]) == 3
    assert data["initial_merit"]["total"] == 1.0
    assert data["final_merit"]["shape"] == 0.2
