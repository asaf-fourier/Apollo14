"""Headline numbers for the pupil report.

Computes a small set of summary scalars (eyebox efficiency, mean cell
brightness, per-mirror average reflectance, stack transparency) from the
manifest + response arrays, and renders them as a compact HTML table
that goes at the top of the Summary page.

All averages over wavelength are uniform (mean over the trace grid)
unless noted — so users can read them as "spectrum-flat" quantities and
apply their own weighting if needed. The eyebox efficiency is
radiometric (energy-fraction reaching the eyebox), since ``response``
already has the projector spectrum baked in.
"""

from __future__ import annotations

import numpy as np


def _projector_spectrum_on(grid_nm: np.ndarray, manifest: dict) -> np.ndarray | None:
    """Interpolate the projector spectrum onto ``grid_nm``.

    Manifest stores wavelengths in apollo14 internal length units; the
    response arrays carry ``wavelengths_nm`` already in nanometers, so
    we convert before interpolating.
    """
    proj = manifest.get("projector") or {}
    spec = proj.get("spectrum")
    if spec is None:
        return None
    wls_nm = np.asarray(spec["wavelengths"]) / 1e-6  # internal units → nm
    radiance = np.asarray(spec["radiance"])
    return np.interp(np.asarray(grid_nm), wls_nm, radiance)


def _mirrors(manifest: dict):
    elements = manifest.get("system", {}).get("elements", [])
    return [
        (e["name"],
         np.asarray(e["wavelengths"]),
         np.asarray(e["reflectance"]))
        for e in elements
        if e.get("type") in ("PartialMirror", "GaussianMirror")
    ]


def _mirror_spacings_mm(manifest: dict) -> list[dict]:
    """Perpendicular distance between consecutive mirror pairs.

    Spacing is ``|normal · (pos_{i+1} − pos_i)|`` — independent of how
    the mirrors are tilted, so the headline doesn't have to know anything
    about the chassis geometry. Positions in the manifest are already in
    mm (``apollo14.units.mm == 1.0``).
    """
    elements = manifest.get("system", {}).get("elements", [])
    mirrors = [e for e in elements
               if e.get("type") in ("PartialMirror", "GaussianMirror")]
    spacings = []
    for prev, curr in zip(mirrors, mirrors[1:]):
        prev_pos = np.asarray(prev["position"])
        curr_pos = np.asarray(curr["position"])
        normal = np.asarray(prev["normal"])
        distance = abs(float(np.dot(normal, curr_pos - prev_pos)))
        spacings.append({
            "from": prev["name"],
            "to": curr["name"],
            "distance_mm": distance,
        })
    return spacings


def compute_headline_numbers(
    manifest: dict,
    response: np.ndarray,
    wavelengths_nm: np.ndarray | None,
) -> dict:
    """Return a dict of headline scalars. Values are ``None`` when the
    inputs needed to compute them are missing (e.g., no projector spectrum
    in the manifest)."""
    numbers: dict = {}

    # ── Mean cell brightness (radiometric, fraction of one direction's flux) ──
    # ``response[s, a, k]`` is per-cell-per-angle radiance with the
    # projector spectrum baked in. Sum over k is per-(cell, angle)
    # radiance; mean over angles is per-cell mean radiance.
    cell_radiance = response.sum(axis=-1).mean(axis=-1)  # (S,)
    mean_cell_radiance = float(cell_radiance.mean())
    min_cell_radiance = float(cell_radiance.min())
    max_cell_radiance = float(cell_radiance.max())

    # ── Eyebox efficiency: total deposited energy / total emitted ────────────
    proj = manifest.get("projector") or {}
    num_rays = int(proj.get("nx", 0)) * int(proj.get("ny", 0))
    num_directions = int(response.shape[1])

    spectrum_on_trace = (None if wavelengths_nm is None
                         else _projector_spectrum_on(wavelengths_nm, manifest))
    if spectrum_on_trace is not None and num_rays > 0:
        per_direction_input = num_rays * float(spectrum_on_trace.sum())
        total_input = num_directions * per_direction_input
        total_eyebox = float(response.sum())
        numbers["eyebox_efficiency_pct"] = 100.0 * total_eyebox / total_input
        numbers["mean_cell_brightness_pct"] = (
            100.0 * mean_cell_radiance / per_direction_input)
        numbers["min_cell_brightness_pct"] = (
            100.0 * min_cell_radiance / per_direction_input)
        numbers["max_cell_brightness_pct"] = (
            100.0 * max_cell_radiance / per_direction_input)
    else:
        numbers["eyebox_efficiency_pct"] = None
        numbers["mean_cell_brightness_pct"] = None
        numbers["min_cell_brightness_pct"] = None
        numbers["max_cell_brightness_pct"] = None

    # ── Per-mirror reflectance + ambient transparency ──────────────────────
    # Ambient light from the world reaches the pupil through *one*
    # mirror's segment of the chassis (each mirror covers a different
    # piece of the see-through aperture), not through the cascade. So
    # the relevant transparency is the per-mirror reflectance averaged
    # across mirrors, not Π_j (1 − r_j). The cascade product is what
    # projector light experiences — see the "Spectrum incident on each
    # mirror" figure.
    mirrors = _mirrors(manifest)
    if mirrors:
        per_mirror = [
            {"name": name, "avg_reflectance": float(np.mean(reflectance))}
            for name, _, reflectance in mirrors
        ]
        avg_r = float(np.mean([m["avg_reflectance"] for m in per_mirror]))
        numbers["per_mirror_avg_reflectance"] = per_mirror
        numbers["avg_per_mirror_reflectance"] = avg_r
        numbers["avg_ambient_transparency"] = 1.0 - avg_r
    else:
        numbers["per_mirror_avg_reflectance"] = []
        numbers["avg_per_mirror_reflectance"] = None
        numbers["avg_ambient_transparency"] = None

    numbers["mirror_spacings_mm"] = _mirror_spacings_mm(manifest)

    return numbers


def _fmt_pct(x: float | None) -> str:
    return "—" if x is None else f"{x:.2f}%"


def _fmt_frac(x: float | None) -> str:
    return "—" if x is None else f"{x:.4f}"


def headline_numbers_html(numbers: dict) -> str:
    """Render the headline numbers dict as an HTML block."""
    eyebox = _fmt_pct(numbers.get("eyebox_efficiency_pct"))
    mean_b = _fmt_pct(numbers.get("mean_cell_brightness_pct"))
    min_b = _fmt_pct(numbers.get("min_cell_brightness_pct"))
    max_b = _fmt_pct(numbers.get("max_cell_brightness_pct"))
    avg_r = _fmt_pct(
        None if numbers.get("avg_per_mirror_reflectance") is None
        else 100.0 * numbers["avg_per_mirror_reflectance"])
    avg_t = _fmt_pct(
        None if numbers.get("avg_ambient_transparency") is None
        else 100.0 * numbers["avg_ambient_transparency"])

    rows = "".join(
        f"<tr><td>{m['name']}-</td>"
        f"<td>{_fmt_pct(100.0 * m['avg_reflectance'])}</td></tr>"
        for m in numbers.get("per_mirror_avg_reflectance", [])
    )

    spacing_rows = "".join(
        f"<tr><td>{s['from']} → {s['to']}</td>"
        f"<td>{s['distance_mm']:.3f} mm</td></tr>"
        for s in numbers.get("mirror_spacings_mm", [])
    )

    return (
        "<div class='headline'>"
        "<h3 style='margin-top:0;'>Headline numbers</h3>"
        "<table class='headline-table'>"
        "<tr><th>Eyebox efficiency (radiometric)</th>"
        f"<td>{eyebox}</td></tr>"
        "<tr><th>Mean cell brightness (per-direction flux)</th>"
        f"<td>{mean_b}</td></tr>"
        "<tr><th>Min / max cell brightness</th>"
        f"<td>{min_b} / {max_b}</td></tr>"
        "<tr><th>Avg per-mirror reflectance (λ-avg)</th>"
        f"<td>{avg_r}</td></tr>"
        "<tr><th>Avg ambient transparency (≈ 1 − avg reflectance)</th>"
        f"<td>{avg_t}</td></tr>"
        "</table>"
        "<table class='headline-table' style='margin-top:.6em;'>"
        "<tr><th>Mirror</th><th>λ-avg reflectance</th></tr>"
        f"{rows}"
        "</table>"
        "<table class='headline-table' style='margin-top:.6em;'>"
        "<tr><th>Mirror pair</th><th>Perpendicular distance</th></tr>"
        f"{spacing_rows}"
        "</table>"
        "</div>"
    )
