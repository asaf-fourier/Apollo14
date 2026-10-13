"""Tests for the portable physical coating-stack report."""

import pytest

from apollo14.export.zemax.reporting.html_plotting import REPORT_LAYOUT_VERSION
from apollo14.export.zemax.reporting.position_space_analysis import write_position_space_index
from apollo14.export.zemax.reporting.zemax_coating_report import (
    normalize_coating_report,
    write_coating_report,
)


def _report_spec():
    return {
        "report_name": "zemax_coating_report",
        "title": "Perseus mirror coating stacks",
        "coating_file": "APOLLO14_COATINGS.DAT",
        "mirrors": [
            {
                "index": 0,
                "name": "mirror_0",
                "face_coatings": [
                    {"face": "front", "coating_name": "STACK_M0"},
                ],
                "stack_coating_name": "STACK_M0",
                "layers": [
                    {"material": "moveon_SiO2", "thickness_nm": 62.04919815},
                    {"material": "moveon_TiO2", "thickness_nm": 25.20490646},
                ],
            },
            {
                "index": 1,
                "name": "mirror_1",
                "face_coatings": [
                    {"face": "front", "coating_name": "FLAT_M1"},
                ],
                "stack_coating_name": None,
                "layers": [],
            },
        ],
    }


def test_coating_report_writes_mirror_layer_tables_and_index_link(tmp_path):
    index_record = write_coating_report(_report_spec(), tmp_path)

    assert index_record == {
        "report_name": "zemax_coating_report",
        "title": "Perseus mirror coating stacks",
        "html": "reports/zemax_coating_report/index.html",
        "detail": "Physical layer recipes from APOLLO14_COATINGS.DAT",
    }
    page = (tmp_path / "reports" / "zemax_coating_report" / "index.html").read_text()
    assert f'data-apollo14-report-layout="{REPORT_LAYOUT_VERSION}"' in page
    assert "Mirror 0 — mirror_0" in page
    assert "Mirror 1 — mirror_1" in page
    assert "front: <code>STACK_M0</code>" in page
    assert "<th>Layer</th><th>Thickness (nm)</th><th>Material</th>" in page
    assert "62.05" in page
    assert "25.20" in page
    assert "moveon_SiO2" in page
    assert "moveon_TiO2" in page
    assert "No physical film stack was exported for this mirror." in page
    assert "All Zemax results" not in page
    assert 'href="position_space_index.html"' not in page
    assert 'href="../../../APOLLO14_COATINGS.DAT"' in page

    write_position_space_index(tmp_path, [], derived_reports=[index_record])
    index_page = (tmp_path / "position_space_index.html").read_text()
    assert 'href="reports/zemax_coating_report/index.html" target="report-frame"' in index_page
    assert "Physical layer recipes from APOLLO14_COATINGS.DAT" in index_page


def test_coating_report_requires_layers_only_for_a_named_physical_stack():
    specification = _report_spec()
    specification["mirrors"][0]["stack_coating_name"] = None

    with pytest.raises(ValueError, match="has layers but no physical stack"):
        normalize_coating_report(specification)
