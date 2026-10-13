"""Position-space spectrum and detector response-cube data model.

The cube validates OpticStudio detector samples and coordinates writing through
the focused rendering module. It stays independent of native ``.DDR`` archive
and report-index concerns, which are exposed by the compatibility façade.
"""

import csv
import json
import math
from array import array
from pathlib import Path

try:
    from .html_plotting import nice_nonnegative_scale
    from .position_space_common import (
        CSV_FLOAT_FORMAT,
        DETECTOR_FLUX_UNIT,
        SCHEMA_VERSION,
        SOURCE_WAVENUMBER,
        SPECTRAL_MODE,
        _detector_label,
        _finite_float,
        _fov_angles_deg,
        _pixel_centers,
        _positive_float,
        _positive_int,
        _product,
        _relative_path,
        _source_label,
        _source_sweep_power,
        _unique_indices,
    )
    from .position_space_rendering import (
        _slice_stem,
        _write_csv_grid,
        _write_heatmap_png,
        _write_html_report,
        _write_npz,
    )
except ImportError:  # Standalone copies in a generated Zemax bundle.
    from html_plotting import nice_nonnegative_scale
    from position_space_common import (  # type: ignore[no-redef]
        CSV_FLOAT_FORMAT,
        DETECTOR_FLUX_UNIT,
        SCHEMA_VERSION,
        SOURCE_WAVENUMBER,
        SPECTRAL_MODE,
        _detector_label,
        _finite_float,
        _fov_angles_deg,
        _pixel_centers,
        _positive_float,
        _positive_int,
        _product,
        _relative_path,
        _source_label,
        _source_sweep_power,
        _unique_indices,
    )
    from position_space_rendering import (  # type: ignore[no-redef]
        _slice_stem,
        _write_csv_grid,
        _write_heatmap_png,
        _write_html_report,
        _write_npz,
    )


class SystemSpectrum:
    """Validated system wavelengths and relative weights for one trace.

    OpticStudio source wavenumber zero samples these wavelengths according to
    their relative weights. The weights are kept explicit in every portable
    artifact so a combined detector map is never mistaken for monochromatic
    data.
    """

    def __init__(self, wavelengths_um, weights=None):
        self.wavelengths_um = [float(value) for value in wavelengths_um]
        if not self.wavelengths_um:
            raise ValueError("System spectrum needs at least one wavelength.")
        if any(not math.isfinite(value) or value <= 0.0 for value in self.wavelengths_um):
            raise ValueError("System wavelengths must be finite positive values in micrometres.")

        if weights is None:
            weights = [1.0] * len(self.wavelengths_um)
        self.weights = [float(value) for value in weights]
        if len(self.weights) != len(self.wavelengths_um):
            raise ValueError("System wavelength weights must match the wavelength count.")
        if (
            any(not math.isfinite(value) or value < 0.0 for value in self.weights)
            or math.fsum(self.weights) <= 0.0
        ):
            raise ValueError(
                "System wavelength weights must be finite, non-negative, and "
                "contain at least one positive value."
            )

    @classmethod
    def from_prescription(cls, prescription):
        """Build a spectrum from the prescription's system settings."""
        try:
            settings = prescription["system"]
            wavelengths_um = settings["wavelengths_um"]
        except KeyError as error:
            raise RuntimeError(
                "Prescription has no system.wavelengths_um for position-space analysis."
            ) from error
        weights = settings.get("wavelength_weights")
        try:
            return cls(wavelengths_um, weights)
        except (TypeError, ValueError) as error:
            raise RuntimeError(f"Prescription has an invalid system spectrum: {error}") from error

    @property
    def normalized_weights(self):
        """Return weights normalized to unit sum for clear result metadata."""
        total = math.fsum(self.weights)
        return [value / total for value in self.weights]

    def metadata(self):
        """Return the portable description of the combined trace spectrum."""
        return {
            "mode": SPECTRAL_MODE,
            "source_wavenumber": SOURCE_WAVENUMBER,
            "wavelengths_um": list(self.wavelengths_um),
            "relative_weights": list(self.weights),
            "normalized_weights": self.normalized_weights,
            "description": (
                "OpticStudio source parameter 4 is zero. Each ray randomly uses "
                "one registered system wavelength according to its relative "
                "Wavelength Data Editor weight; every detector map combines the "
                "resulting polychromatic flux."
            ),
        }


def _coerce_spectrum(spectrum):
    """Accept a SystemSpectrum or a wavelength sequence with unit weights."""
    if isinstance(spectrum, SystemSpectrum):
        return spectrum
    return SystemSpectrum(spectrum)


class PositionSpaceCube:
    """Collect one rectangular detector's polychromatic flux for every field.

    Args:
        detector: The detector entry from prescription.json.
        sources: Source entries in the order used by the sweep.
        spectrum: :class:`SystemSpectrum`, or wavelengths that receive unit
            relative weights.

    The cube stores detector flux per pixel, as returned by
    NCE.GetDetectorData(detector, pixel, 0). It is intentionally called a
    response cube rather than radiance: a discrete source direction has no
    angular measure until a caller supplies a field/angular weighting.
    """

    def __init__(self, detector, sources, spectrum):
        self.detector = dict(detector)
        self.sources = [dict(source) for source in sources]
        self.spectrum = _coerce_spectrum(spectrum)

        if not self.sources:
            raise ValueError("Position-space analysis needs at least one source.")

        data = self.detector.get("data", {})
        self.pixels_x = _positive_int(data.get("pixels_x"), "detector pixels_x")
        self.pixels_y = _positive_int(data.get("pixels_y"), "detector pixels_y")
        self.half_width_x_mm = _positive_float(data.get("half_width_x"), "detector half_width_x")
        self.half_width_y_mm = _positive_float(data.get("half_width_y"), "detector half_width_y")

        self._field_by_source_index = _unique_indices(self.sources, "source")

        fov_angle_flags = []
        for source in self.sources:
            fov_angle_flags.append(_fov_angles_deg(source) is not None)
            _source_sweep_power(source)
        if any(fov_angle_flags) and not all(fov_angle_flags):
            raise ValueError(
                "Position-space sources must provide FOV angles for every field "
                "or for none of them."
            )
        self._fov_angles_available = all(fov_angle_flags)
        self._shape = (
            len(self.sources),
            self.pixels_y,
            self.pixels_x,
        )
        self._slice_size = self.pixels_y * self.pixels_x
        self._flux = array("d", [math.nan]) * _product(self._shape)
        self._recorded = bytearray(len(self.sources))

    @property
    def shape(self):
        """Return the flux-array shape: field, y, x."""
        return self._shape

    @property
    def fov_angles_available(self):
        """Whether every field carries an exported FOV angle pair."""
        return self._fov_angles_available

    def iter_flux(self):
        """Iterate over all samples in field/y/x order without copying them."""
        return iter(self._flux)

    @property
    def detector_label(self):
        """Return the human-facing detector name."""
        return _detector_label(self.detector)

    def record(self, source_index, rows):
        """Store one combined polychromatic detector grid for one source."""
        field_index = self.field_index(source_index)
        if self._recorded[field_index]:
            raise ValueError(f"Response already recorded for field {field_index}.")
        if len(rows) != self.pixels_y:
            raise ValueError(
                f"Detector {self.detector_label!r} returned {len(rows)} rows; "
                f"expected {self.pixels_y}."
            )

        offset = self._field_offset(field_index)
        for row_index, row in enumerate(rows):
            if len(row) != self.pixels_x:
                raise ValueError(
                    f"Detector {self.detector_label!r} row {row_index} contains "
                    f"{len(row)} pixels; expected {self.pixels_x}."
                )
            for column_index, value in enumerate(row):
                value = _finite_float(
                    value,
                    f"Detector {self.detector_label!r} flux at row "
                    f"{row_index}, column {column_index}",
                )
                self._flux[offset + row_index * self.pixels_x + column_index] = value

        self._recorded[field_index] = 1

    def assert_complete(self):
        """Raise a clear error when a field trace is missing."""
        missing = [
            _source_label(source)
            for field_index, source in enumerate(self.sources)
            if not self._recorded[field_index]
        ]
        if missing:
            raise RuntimeError(
                "Position-space response is incomplete; missing " + ", ".join(missing) + "."
            )

    def rows(self, field_index):
        """Return one field's detector map as rows in NCE pixel order."""
        offset = self._field_offset(field_index)
        return [
            list(self._flux[offset + row * self.pixels_x : offset + (row + 1) * self.pixels_x])
            for row in range(self.pixels_y)
        ]

    def flat_field(self, field_index):
        """Return one field's detector map flattened in NCE pixel order.

        This small public read interface lets portable derived-report modules
        consume a completed cube without reaching into its private storage.
        """
        offset = self._field_offset(field_index)
        return list(self._flux[offset : offset + self._slice_size])

    def total_flux(self, field_index):
        """Return the detector sum for one polychromatic field trace."""
        offset = self._field_offset(field_index)
        return math.fsum(self._flux[offset : offset + self._slice_size])

    def field_index(self, source_index):
        """Return the field-axis index belonging to an exported source object."""
        try:
            return self._field_by_source_index[int(source_index)]
        except (KeyError, TypeError, ValueError) as error:
            raise KeyError(f"Source {source_index!r} is not part of this response cube.") from error

    def write(self, output_directory, trace_metadata=None):
        """Write NPZ, JSON, CSV slices, PNG maps, and a local HTML viewer.

        The returned mapping uses absolute paths, so a caller can put links to
        several detectors into a top-level manifest.
        """
        self.assert_complete()
        output_directory = Path(output_directory)
        output_directory.mkdir(parents=True, exist_ok=True)
        slices_directory = output_directory / "slices"
        images_directory = output_directory / "images"
        slices_directory.mkdir(exist_ok=True)
        images_directory.mkdir(exist_ok=True)

        max_flux = max(0.0, max(self._flux))
        color_scale = nice_nonnegative_scale(max_flux)
        slice_entries = []
        for field_index, source in enumerate(self.sources):
            stem = _slice_stem(field_index, source)
            csv_path = slices_directory / f"{stem}.csv"
            image_path = images_directory / f"{stem}.png"
            rows = self.rows(field_index)
            _write_csv_grid(csv_path, rows)
            _write_heatmap_png(image_path, rows, color_scale["maximum"])
            slice_entries.append(
                {
                    "field_index": field_index,
                    "source_label": _source_label(source),
                    "spectral_mode": SPECTRAL_MODE,
                    "source_wavenumber": SOURCE_WAVENUMBER,
                    "total_flux": self.total_flux(field_index),
                    "csv": _relative_path(csv_path, output_directory),
                    "image": _relative_path(image_path, output_directory),
                }
            )

        summary_path = output_directory / "position_space_summary.csv"
        self._write_summary(summary_path)
        metadata = self._metadata(
            trace_metadata or {},
            max_flux,
            color_scale,
            slice_entries,
            summary_path,
            output_directory,
        )
        metadata_path = output_directory / "position_space.json"
        metadata_path.write_text(
            json.dumps(metadata, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )

        npz_path = output_directory / "position_space.npz"
        _write_npz(npz_path, self, metadata)

        html_path = output_directory / "position_space.html"
        _write_html_report(html_path, self, slice_entries, color_scale)

        return {
            "detector_label": self.detector_label,
            "detector_index": self.detector.get("index"),
            "shape": list(self.shape),
            "npz": str(npz_path),
            "metadata": str(metadata_path),
            "summary": str(summary_path),
            "html": str(html_path),
        }

    def _field_offset(self, field_index):
        field_index = int(field_index)
        if not 0 <= field_index < len(self.sources):
            raise IndexError(f"Field index {field_index} is outside 0..{len(self.sources) - 1}.")
        return field_index * self._slice_size

    def _write_summary(self, summary_path):
        pixel_area_mm2 = 4.0 * self.half_width_x_mm * self.half_width_y_mm / float(self._slice_size)
        with Path(summary_path).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "field_index",
                    "source_label",
                    "source_object_index",
                    "fov_x_deg",
                    "fov_y_deg",
                    "source_sweep_power",
                    "spectral_mode",
                    "source_wavenumber",
                    "system_wavelengths_um",
                    "system_wavelength_weights",
                    "total_flux",
                    "mean_flux_per_pixel",
                    "mean_flux_per_mm2",
                ]
            )
            for field_index, source in enumerate(self.sources):
                angles = _fov_angles_deg(source)
                total_flux = self.total_flux(field_index)
                writer.writerow(
                    [
                        field_index,
                        _source_label(source),
                        source["index"],
                        "" if angles is None else format(angles[0], ".9g"),
                        "" if angles is None else format(angles[1], ".9g"),
                        format(_source_sweep_power(source), ".9g"),
                        SPECTRAL_MODE,
                        SOURCE_WAVENUMBER,
                        json.dumps(self.spectrum.wavelengths_um),
                        json.dumps(self.spectrum.weights),
                        format(total_flux, CSV_FLOAT_FORMAT),
                        format(total_flux / self._slice_size, CSV_FLOAT_FORMAT),
                        format(total_flux / (self._slice_size * pixel_area_mm2), CSV_FLOAT_FORMAT),
                    ]
                )

    def _metadata(
        self, trace_metadata, max_flux, color_scale, slice_entries, summary_path, output_directory
    ):
        pixel_area_mm2 = 4.0 * self.half_width_x_mm * self.half_width_y_mm / float(self._slice_size)
        fields = []
        for field_index, source in enumerate(self.sources):
            record = {
                "field_index": field_index,
                "source_label": _source_label(source),
                "source_object_index": source["index"],
                "source_sweep_power": _source_sweep_power(source),
                "fov_angles_deg": _fov_angles_deg(source),
                "intended_direction": source.get("intended_direction"),
            }
            fields.append(record)

        arrays = {
            "flux": {
                "shape": list(self.shape),
                "axes": ["field", "pixel_y", "pixel_x"],
                "unit": DETECTOR_FLUX_UNIT,
                "description": (
                    "NCE.GetDetectorData(..., data=0): combined polychromatic "
                    "flux per detector pixel."
                ),
            },
            "system_wavelength_um": {
                "shape": [len(self.spectrum.wavelengths_um)],
                "unit": "um",
                "is_flux_axis": False,
            },
            "system_wavelength_weight": {
                "shape": [len(self.spectrum.weights)],
                "unit": "relative weight",
                "is_flux_axis": False,
            },
            "pixel_x_center_mm": {
                "shape": [self.pixels_x],
                "unit": "mm in the detector's exported local x frame",
            },
            "pixel_y_center_mm": {
                "shape": [self.pixels_y],
                "unit": "mm in the detector's exported local y frame",
            },
        }
        if self._fov_angles_available:
            arrays["fov_x_deg"] = {
                "shape": [len(self.sources)],
                "unit": "deg",
            }
            arrays["fov_y_deg"] = {
                "shape": [len(self.sources)],
                "unit": "deg",
            }
        else:
            arrays["field_index"] = {
                "shape": [len(self.sources)],
                "unit": "zero-based source order; this prescription lacks FOV angles",
            }

        return {
            "schema": "apollo14.position_space",
            "schema_version": SCHEMA_VERSION,
            "detector": {
                "label": self.detector_label,
                "object_index": self.detector.get("index"),
                "pixels_x": self.pixels_x,
                "pixels_y": self.pixels_y,
                "half_width_x_mm": self.half_width_x_mm,
                "half_width_y_mm": self.half_width_y_mm,
                "pixel_area_mm2": pixel_area_mm2,
                "zemax_local_axes_world": self.detector.get("zemax_local_axes_world"),
                "apollo14_local_axes_world": self.detector.get("apollo14_local_axes_world"),
            },
            "quantity": {
                "name": "detector_flux",
                "detector_data_code": 0,
                "description": (
                    "Flux landing in each detector pixel after one independent "
                    "non-sequential trace."
                ),
                "unit": DETECTOR_FLUX_UNIT,
                "normalization": (
                    "Each field map uses only its own source at source_sweep_power. "
                    "OpticStudio applies the system wavelength weights during "
                    "one wavenumber-zero polychromatic trace; there is no separate "
                    "wavelength axis."
                ),
                "combination_rule": (
                    "For a physically defined scene, rescale each non-zero-power "
                    "field map by desired source power / source_sweep_power, apply "
                    "any angular sample weight, then sum over field. A map traced "
                    "at zero power cannot be normalized. Do not apply the spectral "
                    "weights again."
                ),
            },
            "array_order": ["field", "pixel_y", "pixel_x"],
            "arrays": arrays,
            "fields": fields,
            "spectrum": self.spectrum.metadata(),
            "system_wavelength_um": self.spectrum.wavelengths_um,
            "system_wavelength_weight": self.spectrum.weights,
            "pixel_x_center_mm": _pixel_centers(self.half_width_x_mm, self.pixels_x),
            "pixel_y_center_mm": _pixel_centers(self.half_width_y_mm, self.pixels_y),
            "pixel_row_order": (
                "Detector Rectangle pixel 1 is at local (-x, -y); x changes "
                "fastest and then y advances toward +y. The HTML report draws "
                "the first (-y) row at its top, so its vertical display is "
                "flipped relative to local +y."
            ),
            "trace": trace_metadata,
            "display": {
                "global_max_flux": max_flux,
                "color_scale": {
                    **color_scale,
                    "transform": "square_root",
                    "unit": "OpticStudio system source-power units per pixel",
                },
                "description": (
                    "Every PNG uses the same square-root scale from zero to "
                    "color_scale.maximum, so brightness can be compared across "
                    "fields. global_max_flux retains the exact data maximum."
                ),
            },
            "files": {
                "npz": "position_space.npz",
                "metadata": "position_space.json",
                "summary": _relative_path(summary_path, output_directory),
                "html": "position_space.html",
                "slices": slice_entries,
            },
        }
