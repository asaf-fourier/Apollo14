"""Tests for local numeric eyebox field-overlap analysis."""

import csv
import json
import math

import numpy as np
import pytest

from apollo14.export.zemax.reporting.eyebox_overlap_report import (
    EyeboxOverlapReportSpec,
    write_configured_eyebox_overlap_report,
)
from apollo14.export.zemax.reporting.position_space_analysis import (
    PositionSpaceCube,
    SystemSpectrum,
    write_position_space_index,
)
from apollo14.eyebox_overlap import (
    BoundingQuad,
    DenseZoneSettings,
    _intersect_convex_polygons,
    _largest_inscribed_quad,
    _points_in_convex_polygon,
    calculate_eyebox_overlap,
    load_position_space,
    write_eyebox_overlap_report,
)


def _input_data():
    flux = np.array(
        [
            [[0.0, 4.0, 4.0], [0.0, 4.0, 4.0], [0.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [4.0, 4.0, 0.0], [4.0, 4.0, 0.0]],
        ]
    )
    metadata = [
        {"source_label": "left", "fov_angles_deg": [-2.0, 1.0]},
        {"source_label": "right", "fov_angles_deg": [2.0, 1.0]},
    ]
    return flux, np.array([-1.0, 0.0, 1.0]), np.array([-1.0, 0.0, 1.0]), metadata


def test_calculate_eyebox_overlap_computes_field_and_overlap_polygons():
    flux, x_centers, y_centers, metadata = _input_data()

    result = calculate_eyebox_overlap(
        flux,
        x_centers,
        y_centers,
        detector_label="sample",
        field_metadata=metadata,
    )

    first, second = result.fields
    assert first.source_label == "left"
    assert first.fov_x_deg == -2.0
    assert first.polygon is not None
    assert len(first.polygon.vertices_mm) == 4
    assert first.polygon.area_mm2 == pytest.approx(4.0)
    assert second.polygon is not None
    assert result.overlap_polygon is not None
    assert result.overlap_polygon.area_mm2 == pytest.approx(1.0)
    assert result.contained_overlap_quad is not None
    assert result.contained_overlap_quad == _largest_inscribed_quad(result.overlap_polygon)
    assert result.contained_overlap_quad.area_mm2 == pytest.approx(1.0)
    assert np.all(
        _points_in_convex_polygon(
            result.contained_overlap_quad.vertices_mm,
            result.overlap_polygon,
        )
    )


def test_inscribed_quad_follows_a_rotated_numeric_support_and_stays_inside():
    flux = np.zeros((1, 5, 5))
    flux[0, 0, 2] = 4.0
    flux[0, 1, 1] = 4.0
    flux[0, 2, 0] = 4.0

    result = calculate_eyebox_overlap(
        flux,
        [-2.0, -1.0, 0.0, 1.0, 2.0],
        [-2.0, -1.0, 0.0, 1.0, 2.0],
    )

    field = result.fields[0]
    assert field.polygon is not None
    assert len(field.polygon.vertices_mm) == 6
    assert field.polygon.area_mm2 == pytest.approx(5.0)
    assert result.overlap_polygon == field.polygon
    assert result.contained_overlap_quad is not None
    assert result.contained_overlap_quad.area_mm2 == pytest.approx(4.0)
    assert sorted(result.contained_overlap_quad.side_lengths_mm) == pytest.approx(
        [math.sqrt(2.0), math.sqrt(2.0), math.sqrt(8.0), math.sqrt(8.0)]
    )
    assert np.all(
        _points_in_convex_polygon(result.contained_overlap_quad.vertices_mm, field.polygon)
    )


def test_inscribed_quad_uses_a_four_sided_trapezoid_in_a_tapered_field():
    flux = np.zeros((1, 7, 7))
    flux[0, 0:4, :] = 1.0
    flux[0, 4:7, 1:6] = 1.0

    result = calculate_eyebox_overlap(flux, range(7), range(7))

    field = result.fields[0]
    assert result.contained_overlap_quad is not None
    assert result.contained_overlap_quad.area_mm2 == pytest.approx(42.0)
    assert result.contained_overlap_quad.vertices_mm == (
        (-0.5, -0.5),
        (6.5, -0.5),
        (5.5, 6.5),
        (0.5, 6.5),
    )
    assert field.polygon is not None
    assert field.polygon.area_mm2 == pytest.approx(46.0)
    assert np.all(
        _points_in_convex_polygon(result.contained_overlap_quad.vertices_mm, field.polygon)
    )


def test_support_metrics_include_weak_pixels_while_dense_bounds_exclude_them():
    flux = np.zeros((1, 4, 4))
    flux[0, 0, 0] = 10.0
    flux[0, 3, 3] = 1.0e-12

    result = calculate_eyebox_overlap(
        flux,
        [0.0, 1.0, 2.0, 3.0],
        [0.0, 1.0, 2.0, 3.0],
    )

    field = result.fields[0]
    assert field.support_pixel_count == 2
    assert field.dense_zone_pixel_count == 1
    assert field.polygon is not None
    assert (3.5, 3.5) not in field.polygon.vertices_mm


def test_dense_zone_trims_an_isolated_sparse_margin_sample():
    flux = np.zeros((1, 41, 41))
    flux[0, 10:31, 10:31] = 1.0
    flux[0, 2, 38] = 1.0
    centers = np.arange(-20.0, 21.0)

    result = calculate_eyebox_overlap(flux, centers, centers)

    field = result.fields[0]
    assert result.dense_window_pixels == 3
    assert field.support_pixel_count == 442
    assert field.dense_zone_pixel_count < field.support_pixel_count
    assert field.polygon is not None
    assert max(x for x, _ in field.polygon.vertices_mm) <= 10.5
    assert result.contained_overlap_quad is not None
    assert np.all(
        _points_in_convex_polygon(result.contained_overlap_quad.vertices_mm, field.polygon)
    )


def test_dense_zone_settings_are_validated_and_reported():
    settings = DenseZoneSettings(
        neighborhood_fraction_of_short_detector_axis=0.11,
        minimum_local_positive_pixel_fraction=0.6,
        minimum_local_flux_density_fraction_of_field_max=0.2,
    )

    result = calculate_eyebox_overlap(
        np.ones((1, 20, 20)),
        np.arange(20.0),
        np.arange(20.0),
        dense_zone_settings=settings,
    )

    assert result.dense_zone_settings is settings
    assert result.as_dict()["configuration"] == {
        "dense_neighborhood_fraction_of_short_detector_axis": 0.11,
        "dense_neighborhood_pixels": 3,
        "minimum_local_positive_pixel_fraction": 0.6,
        "minimum_local_flux_density_fraction_of_field_max": 0.2,
    }
    with pytest.raises(ValueError, match=r"interval \(0, 1\]"):
        DenseZoneSettings(minimum_local_positive_pixel_fraction=0.0)


def test_overlap_report_spec_rejects_misspelled_configuration_keys():
    with pytest.raises(ValueError, match="Invalid eyebox overlap report keys"):
        EyeboxOverlapReportSpec.from_mapping({"detctor_label": "eyebox"})


def test_quad_intersection_preserves_a_polygon_with_more_than_four_vertices():
    square = BoundingQuad(
        vertices_mm=((-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)),
    )
    diagonal = math.sqrt(2.0)
    diamond = BoundingQuad(
        vertices_mm=((0.0, -diagonal), (diagonal, 0.0), (0.0, diagonal), (-diagonal, 0.0)),
    )

    overlap = _intersect_convex_polygons([square, diamond])

    assert overlap is not None
    assert len(overlap.vertices_mm) == 8
    assert overlap.area_mm2 == pytest.approx(8.0 * (math.sqrt(2.0) - 1.0))


def test_polygon_intersections_can_exist_without_a_shared_supported_pixel():
    flux, x_centers, y_centers, _ = _input_data()
    flux[1, 1, 1] = 0.0

    result = calculate_eyebox_overlap(flux, x_centers, y_centers)

    support_masks = flux > 0.0
    assert not np.any(np.all(support_masks, axis=0))
    assert result.overlap_polygon is not None
    assert len(result.overlap_polygon.vertices_mm) == 3
    assert result.contained_overlap_quad is None


def test_calculation_rejects_a_field_without_positive_flux():
    flux, x_centers, y_centers, _ = _input_data()
    flux[1] = 0.0

    with pytest.raises(ValueError, match="invalid field indices: 1"):
        calculate_eyebox_overlap(flux, x_centers, y_centers)


def test_report_writer_emits_offline_html_and_numeric_artifacts(tmp_path):
    flux, x_centers, y_centers, metadata = _input_data()
    result = calculate_eyebox_overlap(
        flux,
        x_centers,
        y_centers,
        field_metadata=metadata,
    )
    source_directory = tmp_path / "eyebox"
    source_directory.mkdir()
    source_npz = source_directory / "position_space.npz"
    np.savez(source_npz, flux=flux, pixel_x_center_mm=x_centers, pixel_y_center_mm=y_centers)
    image_directory = source_directory / "images"
    image_directory.mkdir()
    (image_directory / "field_000_left.png").write_bytes(b"png")
    (image_directory / "field_001_right.png").write_bytes(b"png")
    source_metadata = source_directory / "position_space.json"
    source_metadata.write_text(
        json.dumps(
            {
                "files": {
                    "slices": [
                        {
                            "field_index": 0,
                            "image": "images/field_000_left.png",
                        },
                        {
                            "field_index": 1,
                            "image": "images/field_001_right.png",
                        },
                    ]
                }
            }
        )
    )
    (source_directory / "position_space.html").write_text("source maps")

    report_directory = tmp_path / "reports" / "eyebox_field_overlap"
    report_directory.mkdir(parents=True)
    legacy_rectangle_csv = report_directory / "field_rectangles.csv"
    legacy_rectangle_csv.write_text("obsolete")
    legacy_field_quad_csv = report_directory / "field_quads.csv"
    legacy_field_quad_csv.write_text("obsolete")
    legacy_quad_overlap_csv = report_directory / "quad_overlap.csv"
    legacy_quad_overlap_csv.write_text("obsolete")
    paths = write_eyebox_overlap_report(
        result,
        report_directory,
        source_npz_path=source_npz,
        source_metadata_path=source_metadata,
    )
    assert not legacy_rectangle_csv.exists()
    assert not legacy_field_quad_csv.exists()
    assert not legacy_quad_overlap_csv.exists()

    payload = json.loads(paths["json"].read_text())
    assert payload["schema"] == "apollo14.eyebox_field_overlap"
    assert payload["schema_version"] == 13
    assert payload["provenance"]["image_processing_used"] is False
    assert payload["provenance"]["support_rule"] == (
        "A pixel belongs to a field when its numeric flux is positive."
    )
    assert payload["provenance"]["coordinate_frame"] == (
        "Vertices are detector-local (x, y) coordinates in millimetres."
    )
    assert "convex hull" in payload["provenance"]["field_polygon_definition"]
    assert "geometric intersection" in payload["provenance"]["overlap_polygon_definition"]
    assert "maximum-area four-vertex" in payload["provenance"]["contained_quad_definition"]
    assert (
        "Top, Bottom, Left, and Right are assigned"
        in payload["provenance"]["quad_vertex_and_side_order"]
    )
    assert payload["configuration"] == {
        "dense_neighborhood_fraction_of_short_detector_axis": 0.05,
        "dense_neighborhood_pixels": 1,
        "minimum_local_positive_pixel_fraction": 0.5,
        "minimum_local_flux_density_fraction_of_field_max": 0.1,
    }
    assert payload["fields"][0]["dense_zone_pixel_count"] == 4
    assert payload["fields"][0]["bounding_polygon"]["vertex_count"] == 4
    assert payload["fields"][0]["bounding_polygon"]["side_lengths_mm"] == pytest.approx(
        [2.0, 2.0, 2.0, 2.0]
    )
    assert "bounding_rectangle" not in payload["fields"][0]
    assert "bounding_quad" not in payload["fields"][0]
    assert payload["overlap_polygon"]["area_mm2"] == pytest.approx(1.0)
    assert payload["contained_overlap_quad"]["area_mm2"] == pytest.approx(1.0)
    assert payload["contained_overlap_quad"]["side_lengths_mm"] == pytest.approx(
        [1.0, 1.0, 1.0, 1.0]
    )
    assert set(payload["contained_overlap_quad"]["side_names"]) == {
        "Top",
        "Bottom",
        "Left",
        "Right",
    }
    assert "bounding_rectangle_intersection" not in payload
    assert "bounding_quad_intersection" not in payload
    assert "all_field_support" not in payload
    assert payload["files"]["source_position_space_html"] == "../../eyebox/position_space.html"
    assert payload["files"]["field_images"] == [
        {
            "field_index": 0,
            "source_label": "left",
            "image": "../../eyebox/images/field_000_left.png",
        },
        {
            "field_index": 1,
            "source_label": "right",
            "image": "../../eyebox/images/field_001_right.png",
        },
    ]
    html = paths["html"].read_text()
    assert 'data-apollo14-report-layout="horizontal-scale-units-v1"' in html
    assert "__REPORT_DATA__" not in html
    assert html.count('viewBox="0 0 760 760"') == 1
    assert "Field bounding polygons, overlap polygon, and contained quad" in html
    assert "Contained quad of the all-field overlap polygon" in html
    assert '<div class="plot-selector-layout">' in html
    assert '<aside class="field-selector-panel"' in html
    assert "grid-template-columns: minmax(520px, 760px) minmax(220px, 1fr)" in html
    assert "<tr><th>Show</th><th>Field</th></tr>" in html
    assert all(
        f'id="{element_id}"' in html
        for element_id in (
            "clearFields",
            "selectFields",
            "showFieldPoints",
            "overlapVertices",
            "overlapSides",
        )
    )
    assert "addFieldPointImages" in html
    assert "const unitScale = Math.min" in html
    assert "sideLabel.textContent = `${format(polygon.side_lengths_mm[index])} mm`" in html
    assert "coordinates.textContent = `(${format(x)}, ${format(y)})`" in html
    assert "const displayOrder = ['Top', 'Bottom', 'Left', 'Right']" in html
    assert "'font-size':18" in html
    assert "<h2>Method</h2>" not in html
    assert "sparse dots can appear outside the density-trimmed polygons" in html
    csv_lines = paths["contained_overlap_quad_csv"].read_text().splitlines()
    assert csv_lines[0] == "vertex,x_mm,y_mm,outgoing_side,side_length_mm"
    assert len(csv_lines) == 5
    with paths["contained_overlap_quad_csv"].open(newline="") as stream:
        csv_rows = list(csv.DictReader(stream))
    assert csv_rows[0]["vertex"] == "V1"
    assert {row["outgoing_side"] for row in csv_rows} == {"Top", "Bottom", "Left", "Right"}
    assert csv_rows[0]["outgoing_side"] == payload["contained_overlap_quad"]["side_names"][0]
    assert float(csv_rows[0]["x_mm"]) == pytest.approx(
        payload["contained_overlap_quad"]["vertices_mm"][0][0]
    )
    assert float(csv_rows[0]["y_mm"]) == pytest.approx(
        payload["contained_overlap_quad"]["vertices_mm"][0][1]
    )
    assert float(csv_rows[0]["side_length_mm"]) == pytest.approx(
        payload["contained_overlap_quad"]["side_lengths_mm"][0]
    )


def test_portable_cube_report_is_linked_from_the_zemax_index(tmp_path):
    detector = {
        "index": 7,
        "comment": "eyebox",
        "data": {
            "pixels_x": 3,
            "pixels_y": 3,
            "half_width_x": 1.5,
            "half_width_y": 1.5,
        },
    }
    sources = [
        {
            "index": 11,
            "label": "left",
            "fov_angles_deg": [-2.0, 1.0],
            "data": {"sweep_power": 1.0},
        },
        {
            "index": 12,
            "label": "right",
            "fov_angles_deg": [2.0, 1.0],
            "data": {"sweep_power": 1.0},
        },
    ]
    flux, _x_centers, _y_centers, _metadata = _input_data()
    cube = PositionSpaceCube(detector, sources, SystemSpectrum([0.55]))
    for source, rows in zip(sources, flux, strict=True):
        cube.record(source["index"], rows.tolist())
    cube.write(tmp_path / "eyebox")

    result = write_configured_eyebox_overlap_report(
        [cube],
        EyeboxOverlapReportSpec(),
        tmp_path,
    )

    assert result is not None
    assert (
        result["html"] == (tmp_path / "reports" / "eyebox_field_overlap" / "index.html").resolve()
    )
    assert result["detector_label"] == "eyebox"
    index_record = {
        "report_name": result["report_name"],
        "title": result["title"],
        "detail": result["detail"],
        "detector_label": result["detector_label"],
        "html": result["html"].relative_to(tmp_path).as_posix(),
        "json": result["json"].relative_to(tmp_path).as_posix(),
        "contained_overlap_quad_csv": result["contained_overlap_quad_csv"]
        .relative_to(tmp_path)
        .as_posix(),
    }
    write_position_space_index(tmp_path, [], derived_reports=[index_record])
    index_html = (tmp_path / "position_space_index.html").read_text()
    assert 'href="reports/eyebox_field_overlap/index.html" target="report-frame"' in index_html
    assert "Eyebox field overlap" in index_html


def test_load_position_space_reads_sibling_metadata_before_embedded_metadata(tmp_path):
    npz_path = tmp_path / "position_space.npz"
    np.savez(
        npz_path,
        flux=np.ones((1, 2, 2)),
        pixel_x_center_mm=np.array([-0.5, 0.5]),
        pixel_y_center_mm=np.array([-0.5, 0.5]),
        **{"metadata.json": json.dumps({"detector": {"label": "embedded"}})},
    )
    (tmp_path / "position_space.json").write_text(json.dumps({"detector": {"label": "sibling"}}))

    loaded = load_position_space(npz_path)

    assert loaded.metadata["detector"]["label"] == "sibling"
    assert loaded.metadata_path == tmp_path / "position_space.json"


def test_calculation_rejects_nonuniform_coordinate_grid():
    with pytest.raises(ValueError, match="uniform detector-pixel spacing"):
        calculate_eyebox_overlap(
            np.ones((1, 2, 3)),
            [-1.0, 0.0, 2.0],
            [-0.5, 0.5],
        )
