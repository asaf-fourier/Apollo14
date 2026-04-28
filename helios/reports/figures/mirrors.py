"""Mirror reflectance curves — one line per mirror across wavelength.

Loaded directly from the run manifest (no tracing needed). Reveals the
optimizer's reflectance design at a glance: did mirror 0 collapse to
flat? did mirror 3 specialize on red? are the per-color Gaussians
where you'd expect them?
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from apollo14.units import nm


def mirror_reflectance_figure(manifest: dict) -> go.Figure:
    """Plot ``r_m(λ)`` for every mirror element in the manifest.

    Reads ``manifest["system"]["elements"]`` and pulls the ``wavelengths``
    + ``reflectance`` arrays from any element whose type is
    ``PartialMirror`` or ``GaussianMirror``.
    """
    elements = manifest.get("system", {}).get("elements", [])
    mirrors = [
        (e["name"], np.asarray(e["wavelengths"]), np.asarray(e["reflectance"]))
        for e in elements
        if e.get("type") in ("PartialMirror", "GaussianMirror")
    ]

    fig = go.Figure()
    if not mirrors:
        fig.update_layout(
            title="Mirror reflectance curves — no mirrors found in manifest",
            width=700, height=420,
        )
        return fig

    for name, wls_internal, reflectance in mirrors:
        wls_nm = np.asarray(wls_internal) / nm
        fig.add_trace(go.Scatter(
            x=wls_nm.tolist(),
            y=np.asarray(reflectance).tolist(),
            mode="lines+markers",
            name=name,
        ))

    fig.update_layout(
        title="Mirror reflectance curves",
        xaxis_title="wavelength (nm)",
        yaxis_title="reflectance",
        yaxis=dict(rangemode="tozero"),
        width=700, height=420,
        margin=dict(l=60, r=20, t=60, b=50),
    )
    return fig
