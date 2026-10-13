"""Native OpticStudio detector-state archive and ZPL viewer launcher.

This module plans and indexes ``.DDR`` files saved by the generated sweep. It
also writes the manifest and ZPL macro that restore one field in OpticStudio;
portable NPZ/HTML rendering remains in separate modules.
"""

import json
from pathlib import Path

try:
    from .position_space_common import (
        NATIVE_DETECTOR_ARCHIVE_SCHEMA_VERSION,
        SOURCE_WAVENUMBER,
        SPECTRAL_MODE,
        _detector_label,
        _fov_angles_deg,
        _positive_int,
        _relative_path_from,
        _source_label,
        _source_sweep_power,
        _unique_indices,
        _zpl_comment_text,
        _zpl_string,
    )
    from .position_space_core import _coerce_spectrum
    from .position_space_rendering import _format_angle_deg
except ImportError:  # Standalone copies in a generated Zemax bundle.
    from position_space_common import (  # type: ignore[no-redef]
        NATIVE_DETECTOR_ARCHIVE_SCHEMA_VERSION,
        SOURCE_WAVENUMBER,
        SPECTRAL_MODE,
        _detector_label,
        _fov_angles_deg,
        _positive_int,
        _relative_path_from,
        _source_label,
        _source_sweep_power,
        _unique_indices,
        _zpl_comment_text,
        _zpl_string,
    )
    from position_space_core import _coerce_spectrum
    from position_space_rendering import _format_angle_deg


class NativeDetectorArchive:
    """Persist each polychromatic field as native detector data and a ZPL viewer.

    A Detector Rectangle's ``.DDR`` file is OpticStudio's own storage format.
    This class only plans and indexes those files; the ZOS-API runner is
    responsible for calling ``NCE.SaveDetector`` immediately after each trace.
    The generated macro later calls ``LOADDETECTOR`` to restore a selected
    field state without changing source settings or tracing again.

    Args:
        output_directory: Directory containing the normal position-space output.
        model_directory: Directory holding the matching ``.zmx`` model. The ZPL
            macro uses paths relative to that directory, because OpticStudio
            resolves ``LOADDETECTOR`` paths relative to the open lens file.
        sources: Exported sources in field-axis order.
        detectors: Exported Detector Rectangle entries.
        spectrum: The system spectrum combined in every saved field state.
        macro_name: Filename for the generated OpticStudio macro.
    """

    def __init__(
        self,
        output_directory,
        model_directory,
        sources,
        detectors,
        spectrum,
        macro_name="apollo14_view_position_space.zpl",
    ):
        self.output_directory = Path(output_directory).resolve()
        self.model_directory = Path(model_directory).resolve()
        self.archive_directory = self.output_directory / "zemax_detector_data"
        self.macro_path = self.model_directory / macro_name
        self.sources = [dict(source) for source in sources]
        self.detectors = [dict(detector) for detector in detectors]
        self.spectrum = _coerce_spectrum(spectrum)
        self._field_by_source_index = _unique_indices(self.sources, "source")
        self._detector_by_index = {
            _positive_int(detector.get("index"), "detector index"): detector
            for detector in self.detectors
        }
        if len(self._detector_by_index) != len(self.detectors):
            raise ValueError("Duplicate detector index in native detector archive.")
        if not self.sources:
            raise ValueError("Native detector archive needs at least one source.")
        if not self.detectors:
            raise ValueError("Native detector archive needs at least one detector.")
        self._records = {}

    def data_path(self, detector_index, source_index):
        """Return the deterministic ``.DDR`` location for one field trace."""
        detector_index = _positive_int(detector_index, "detector index")
        try:
            self._detector_by_index[detector_index]
        except KeyError as error:
            raise KeyError(
                f"Detector {detector_index} is not part of this native archive."
            ) from error
        field_index = self.field_index(source_index)
        return (
            self.archive_directory / f"detector_{detector_index}" / f"field_{field_index + 1}.DDR"
        )

    def field_index(self, source_index):
        """Return a zero-based field index for one source object number."""
        try:
            return self._field_by_source_index[int(source_index)]
        except (KeyError, TypeError, ValueError) as error:
            raise KeyError(
                f"Source {source_index!r} is not part of this native archive."
            ) from error

    def record(self, detector_index, source_index, path):
        """Register one successfully written native detector-data file."""
        detector_index = _positive_int(detector_index, "detector index")
        field_index = self.field_index(source_index)
        expected_path = self.data_path(detector_index, source_index).resolve()
        actual_path = Path(path).resolve()
        if actual_path != expected_path:
            raise ValueError(
                f"Native detector file path {actual_path} does not match expected "
                f"path {expected_path}."
            )
        key = detector_index, field_index
        if key in self._records:
            raise ValueError(
                "Native detector data already recorded for detector "
                f"{detector_index}, field {field_index}."
            )
        self._records[key] = actual_path

    def write(self, viewer_config_path=None):
        """Write a manifest and a one-macro native Detector Viewer launcher."""
        self._assert_complete()
        self.archive_directory.mkdir(parents=True, exist_ok=True)
        viewer_config_path = (
            None if viewer_config_path is None else Path(viewer_config_path).resolve()
        )
        manifest_path = self.archive_directory / "native_detector_archive.json"
        macro_text = self._zpl_macro_text(viewer_config_path)
        self.macro_path.write_text(macro_text, encoding="utf-8")
        payload = self._manifest(viewer_config_path, manifest_path)
        manifest_path.write_text(
            json.dumps(payload, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        return {
            "manifest": str(manifest_path),
            "macro": str(self.macro_path),
            "archive_directory": str(self.archive_directory),
            "viewer_config": (None if viewer_config_path is None else str(viewer_config_path)),
            "records": len(self._records),
        }

    def _assert_complete(self):
        missing = []
        for detector in self.detectors:
            detector_index = int(detector["index"])
            for field_index, _source in enumerate(self.sources):
                key = detector_index, field_index
                if key not in self._records:
                    missing.append(f"detector {detector_index}, field {field_index + 1}")
        if missing:
            raise RuntimeError(
                "Native detector archive is incomplete; missing " + "; ".join(missing) + "."
            )

    def _manifest(self, viewer_config_path, manifest_path):
        return {
            "schema": "apollo14.native_detector_archive",
            "schema_version": NATIVE_DETECTOR_ARCHIVE_SCHEMA_VERSION,
            "description": (
                "One polychromatic OpticStudio .DDR file per detector and "
                "field. Load them with the generated ZPL macro rather than "
                "retracing a selected field response."
            ),
            "format": {
                "extension": ".DDR",
                "producer": "ZOSAPI INonSeqEditor.SaveDetector",
                "quantity": (
                    "Native Detector Rectangle state, including the traced "
                    "incoherent position-space flux."
                ),
            },
            "fields": [
                {
                    "field_number": field_index + 1,
                    "field_index_zero_based": field_index,
                    "source_label": _source_label(source),
                    "source_object_index": source["index"],
                    "source_sweep_power": _source_sweep_power(source),
                    "fov_angles_deg": _fov_angles_deg(source),
                }
                for field_index, source in enumerate(self.sources)
            ],
            "spectrum": self.spectrum.metadata(),
            "detectors": [
                {
                    "object_index": detector["index"],
                    "label": _detector_label(detector),
                }
                for detector in self.detectors
            ],
            "files": [
                self._record_manifest_entry(detector, field_index)
                for detector in self.detectors
                for field_index in range(len(self.sources))
            ],
            "viewer": {
                "macro": _relative_path_from(self.macro_path, self.model_directory),
                "settings": (
                    None
                    if viewer_config_path is None
                    else _relative_path_from(viewer_config_path, self.model_directory)
                ),
                "usage": (
                    "Open the matching .zmx in OpticStudio, run the macro, "
                    "and enter the one-based field number."
                ),
                "native_display": (
                    "Detector Viewer displays Incoherent Irradiance, which is "
                    "the saved flux per pixel divided by the constant pixel area."
                ),
            },
            "manifest": _relative_path_from(manifest_path, self.model_directory),
        }

    def _record_manifest_entry(self, detector, field_index):
        detector_index = int(detector["index"])
        path = self._records[detector_index, field_index]
        source = self.sources[field_index]
        return {
            "detector_object_index": detector_index,
            "detector_label": _detector_label(detector),
            "field_number": field_index + 1,
            "field_index_zero_based": field_index,
            "source_label": _source_label(source),
            "source_object_index": source["index"],
            "source_sweep_power": _source_sweep_power(source),
            "spectral_mode": SPECTRAL_MODE,
            "source_wavenumber": SOURCE_WAVENUMBER,
            "file": _relative_path_from(path, self.model_directory),
        }

    def _zpl_macro_text(self, viewer_config_path):
        field_lines = [
            "! Generated by Apollo14. Regenerate this bundle rather than editing it.",
            "! Open the matching apollo14_combiner.zmx before running this macro.",
            "! The macro restores saved .DDR detector data; it does not trace rays.",
            "",
            'PRINT "Apollo14 native position-space result viewer"',
            'PRINT "Field numbers:"',
        ]
        for field_index, source in enumerate(self.sources, start=1):
            angles = _fov_angles_deg(source)
            details = []
            if angles is not None:
                details.append(
                    f"FOV x={_format_angle_deg(angles[0])} deg, "
                    f"y={_format_angle_deg(angles[1])} deg"
                )
            details.append(f"sweep power={_source_sweep_power(source):.9g}")
            suffix = " (" + "; ".join(details) + ")"
            field_lines.append(
                f'PRINT "  {field_index}: {_zpl_comment_text(_source_label(source))}'
                f'{_zpl_comment_text(suffix)}"'
            )
        field_lines.append(
            'PRINT "Spectrum: polychromatic System Wavelengths (source wavenumber 0)"'
        )
        for spectrum_index, wavelength_um in enumerate(self.spectrum.wavelengths_um):
            weight = self.spectrum.weights[spectrum_index]
            field_lines.append(f'PRINT "  {wavelength_um:.9g} um, relative weight {weight:.9g}"')
        field_lines.extend(
            [
                "",
                "field = 0",
                f'INPUT "Field number (1-{len(self.sources)})", field',
                "IF (field < 1)",
                '  PRINT "Field number is out of range."',
                "  END",
                "ENDIF",
                f"IF (field > {len(self.sources)})",
                '  PRINT "Field number is out of range."',
                "  END",
                "ENDIF",
                "",
            ]
        )
        for field_index in range(len(self.sources)):
            field_lines.append(f"IF (field == {field_index + 1})")
            for detector in self.detectors:
                path = self._records[int(detector["index"]), field_index]
                field_lines.append(
                    "  LOADDETECTOR 1, {detector_index}, {path}".format(
                        detector_index=detector["index"],
                        path=_zpl_string(_relative_path_from(path, self.model_directory)),
                    )
                )
            field_lines.append("ENDIF")
        field_lines.extend([""])
        field_lines.extend(self._zpl_viewer_lines(viewer_config_path))
        field_lines.extend(["END", ""])
        return "\n".join(field_lines)

    def _zpl_viewer_lines(self, viewer_config_path):
        """Return the Detector Viewer launch commands for the ZPL macro.

        A current ZOS-API installation supplies a saved, lens-local ``.cfg``
        file.  If that save is unavailable, the macro derives a private config
        from OpticStudio's stock Detector Viewer config.  It never changes the
        user's global Detector Viewer defaults.
        """
        if viewer_config_path is not None:
            return [
                'PRINT "Loaded native detector data. Opening False Color Detector Viewer."',
                'OPENANALYSISWINDOW "DVR", '
                + _zpl_string(_relative_path_from(viewer_config_path, self.model_directory)),
            ]

        first_detector = int(self.detectors[0]["index"])
        fallback_config_path = self.archive_directory / "apollo14_detector_viewer_false_color.cfg"
        relative_config_path = _zpl_string(
            _relative_path_from(fallback_config_path, self.model_directory)
        )
        return [
            'PRINT "Loaded native detector data. Opening False Color Detector Viewer."',
            "lens_path$ = $PATHNAME()",
            "data_path$ = $DATAPATH()",
            r'default_viewer_config$ = data_path$ + "\Configs\DVR.CFG"',
            f"viewer_config$ = lens_path$ + {relative_config_path}",
            "COPYFILE default_viewer_config$, viewer_config$",
            'MODIFYSETTINGS viewer_config$, DVW_SURFACE, 1, "DVR"',
            (f'MODIFYSETTINGS viewer_config$, DVW_DETECTOR, {first_detector}, "DVR"'),
            'MODIFYSETTINGS viewer_config$, DVW_SHOW, 2, "DVR"',
            'MODIFYSETTINGS viewer_config$, DVW_DATA, 0, "DVR"',
            'MODIFYSETTINGS viewer_config$, DVW_SCALE, 0, "DVR"',
            'MODIFYSETTINGS viewer_config$, DVW_SMOOTHING, 0, "DVR"',
            'OPENANALYSISWINDOW "DVR", viewer_config$',
        ]
