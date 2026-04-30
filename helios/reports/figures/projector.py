"""Projector spectrum and per-mirror residual incident spectrum.

Two figures, both read straight from the run manifest (no tracing):

- :func:`projector_spectrum_figure` — what the projector emits.
- :func:`mirror_input_spectrum_figure` — what each mirror sees on its
  way down the stack: the projector spectrum after upstream mirrors
  have transmitted ``(1 − r_j(λ))`` of the light through.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from apollo14.units import nm


def _projector_spectrum(manifest):
    """Return (wavelengths_nm, radiance) or ``None`` if missing."""
    spec = manifest.get("projector", {}).get("spectrum") if manifest.get("projector") else None
    if spec is None:
        return None
    return np.asarray(spec["wavelengths"]) / nm, np.asarray(spec["radiance"])


def _mirrors(manifest):
    """Return ``[(name, wavelengths_nm, reflectance), ...]`` in stack order."""
    elements = manifest.get("system", {}).get("elements", [])
    return [
        (e["name"], np.asarray(e["wavelengths"]) / nm, np.asarray(e["reflectance"]))
        for e in elements
        if e.get("type") in ("PartialMirror", "GaussianMirror")
    ]


def _empty(title: str) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(title=title, width=700, height=420)
    return fig


def projector_spectrum_figure(manifest: dict) -> go.Figure:
    """Plot the projector's emitted spectrum (peak-normalized)."""
    spectrum = _projector_spectrum(manifest)
    if spectrum is None:
        return _empty("Projector spectrum — no spectrum found in manifest")
    wls_nm, radiance = spectrum
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=wls_nm.tolist(), y=radiance.tolist(),
        mode="lines", name="projector",
    ))
    fig.update_layout(
        title="Projector emitted spectrum",
        xaxis_title="wavelength (nm)",
        yaxis_title="relative radiance",
        yaxis=dict(rangemode="tozero"),
        width=700, height=420,
        margin=dict(l=60, r=20, t=60, b=50),
    )
    return fig


def mirror_input_spectrum_figure(manifest: dict) -> go.Figure:
    """Plot the cumulative-transmission spectrum incident on each mirror.

    Mirror 0 sees the full projector. Mirror 1 sees what mirror 0
    transmitted (``projector × (1 − r_0)``). Mirror i sees
    ``projector × Π_{j<i} (1 − r_j)``. A trailing dashed curve shows
    what's left after the last mirror — the light exiting the stack.
    """
    spectrum = _projector_spectrum(manifest)
    mirrors = _mirrors(manifest)
    if spectrum is None or not mirrors:
        return _empty("Mirror input spectra — projector or mirrors missing")

    proj_wls_nm, proj_radiance = spectrum
    # Mirrors share a wavelength grid (the optimization's spectral band);
    # interpolate the projector onto it.
    _, mirror_wls_nm, _ = mirrors[0]
    incident = np.interp(mirror_wls_nm, proj_wls_nm, proj_radiance)

    fig = go.Figure()
    for name, _, reflectance in mirrors:
        fig.add_trace(go.Scatter(
            x=mirror_wls_nm.tolist(), y=incident.tolist(),
            mode="lines", name=f"input to {name}",
        ))
        incident = incident * (1.0 - np.asarray(reflectance))

    fig.add_trace(go.Scatter(
        x=mirror_wls_nm.tolist(), y=incident.tolist(),
        mode="lines", name="exiting stack",
        line=dict(dash="dash", color="black"),
    ))

    fig.update_layout(
        title="Spectrum incident on each mirror (cumulative transmission)",
        xaxis_title="wavelength (nm)",
        yaxis_title="relative radiance",
        yaxis=dict(rangemode="tozero"),
        width=700, height=420,
        margin=dict(l=60, r=20, t=60, b=50),
    )
    return fig
