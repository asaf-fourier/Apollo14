"""Portable report generation copied into each Zemax export bundle.

The package is organized by responsibility:

* ``position_space_*`` owns detector-cube storage, plots, and the report index.
* ``eyebox_overlap_geometry`` owns dependency-free convex geometry.
* ``eyebox_overlap_core`` turns per-field flux into dense polygons and overlap.
* ``eyebox_overlap_report`` adapts completed cubes and writes report artifacts.
* ``eyebox_overlap_report_template`` contains the offline HTML presentation.
* ``zemax_*`` modules generate the remaining Zemax-side analyses.

Modules listed by :data:`apollo14.export.zemax.bundle.PORTABLE_REPORT_MODULES`
must remain standard-library-only and compatible with the Zemax Python 3.8
runtime.
"""
