"""Export an Apollo14 optical system to third-party optical design tools.

Currently targets **Zemax OpticStudio** non-sequential mode, for independent
validation of a design Apollo14 has traced and Helios has optimized. The
bundle is regenerable from the same source of truth the tracer uses
(:mod:`apollo14.perseus`), so the Zemax model cannot silently drift from the
optimizer.

``apollo14.export`` never imports ``helios`` — callers pass optimizer output
in as plain arrays (see ``examples/export_perseus_zemax.py``).
"""

from apollo14.export.bundle import export_zemax_bundle

__all__ = ["export_zemax_bundle"]
