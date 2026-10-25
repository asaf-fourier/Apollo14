"""Stable public interface for portable Zemax position-space analysis.

Implementation is split by responsibility so detector-cube calculations, file
rendering, report navigation, and native OpticStudio archives can evolve
independently. This façade preserves the original imports used by generated
scripts and existing callers. All dependency modules are copied beside it in a
Zemax bundle.
"""

try:
    from .eyebox_overlap_report import write_configured_eyebox_overlap_report
    from .html_plotting import format_compact_number
    from .position_space_common import (
        CSV_FLOAT_FORMAT,
        DETECTOR_FLUX_UNIT,
        NATIVE_DETECTOR_ARCHIVE_SCHEMA_VERSION,
        SCHEMA_VERSION,
        SOURCE_WAVENUMBER,
        SPECTRAL_MODE,
    )
    from .position_space_core import PositionSpaceCube, SystemSpectrum
    from .position_space_index import detector_directory_name, write_position_space_index
    from .position_space_native import NativeDetectorArchive
    from .position_space_rendering import _field_display, _format_angle_deg
except ImportError:  # Standalone copies in a generated Zemax bundle.
    from eyebox_overlap_report import write_configured_eyebox_overlap_report
    from html_plotting import format_compact_number
    from position_space_common import (  # type: ignore[no-redef]
        CSV_FLOAT_FORMAT,
        DETECTOR_FLUX_UNIT,
        NATIVE_DETECTOR_ARCHIVE_SCHEMA_VERSION,
        SCHEMA_VERSION,
        SOURCE_WAVENUMBER,
        SPECTRAL_MODE,
    )
    from position_space_core import PositionSpaceCube, SystemSpectrum
    from position_space_index import detector_directory_name, write_position_space_index
    from position_space_native import NativeDetectorArchive
    from position_space_rendering import _field_display, _format_angle_deg

__all__ = [
    "CSV_FLOAT_FORMAT",
    "DETECTOR_FLUX_UNIT",
    "NATIVE_DETECTOR_ARCHIVE_SCHEMA_VERSION",
    "SCHEMA_VERSION",
    "SOURCE_WAVENUMBER",
    "SPECTRAL_MODE",
    "NativeDetectorArchive",
    "PositionSpaceCube",
    "SystemSpectrum",
    "_field_display",
    "_format_angle_deg",
    "detector_directory_name",
    "format_compact_number",
    "write_configured_eyebox_overlap_report",
    "write_position_space_index",
]
