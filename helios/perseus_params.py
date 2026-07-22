"""Parametrized Perseus combiner — design variables for optimization.

The differentiable counterpart to :func:`apollo14.perseus.build_perseus_system`.
It exposes the Perseus geometry as a function of :class:`CombinerParams`
(per-mirror reflectance curves + inter-mirror spacings), so a merit
function can be differentiated straight back to those design variables —
exactly like :func:`helios.combiner_params.build_parametrized_system` does
for Talos, but on the tilted 8-mirror Perseus geometry.

The rigid-body tilts (``pantoscopic_tilt``, ``projector_tilt``) and the
chassis thickness are **build arguments**, not design variables — they set
the frame the optimizer works inside. Geometry placement is delegated to
:func:`apollo14.perseus.build_perseus_geometry` so this module and the
reference builder never drift.
"""

from __future__ import annotations

import jax.numpy as jnp

from apollo14.elements.partial_mirror import PartialMirror
from apollo14.materials import air
from apollo14.perseus import (
    PERSEUS_CHASSIS_Z,
    PERSEUS_PANTOSCOPIC_TILT,
    PERSEUS_PROJECTOR_TILT,
    build_perseus_geometry,
)
from apollo14.system import OpticalSystem
from apollo14.units import mm, nm
from helios.combiner_params import CombinerParams

# Fallback probe grid for curves that carry no wavelength centers of their
# own (e.g. a flat ConstantCurve). Spans the visible band densely so the
# mirror's stored reflectance interpolates cleanly at any trace wavelength.
_DEFAULT_PROBE_WAVELENGTHS = jnp.linspace(420 * nm, 660 * nm, 100)


def build_parametrized_perseus(
    params: CombinerParams,
    *,
    probe_wavelengths: jnp.ndarray | None = None,
    pantoscopic_tilt: float = PERSEUS_PANTOSCOPIC_TILT,
    projector_tilt: float = PERSEUS_PROJECTOR_TILT,
    chassis_z: float = PERSEUS_CHASSIS_Z,
    **geometry_kwargs,
) -> OpticalSystem:
    """Build the Perseus system with ``params`` as the design variables.

    Args:
        params: :class:`CombinerParams` — ``spacings`` ``(M-1,)`` and a
            batched reflectance ``curves`` with leaves shaped ``(M, …)``.
            Both flow into the returned system, so gradients reach them.
        probe_wavelengths: ``(K,)`` grid on which each mirror's curve is
            eager-sampled before storage. Defaults to a dense visible-band
            grid (works for curves without their own centers, e.g. a flat
            :class:`~apollo14.spectral.ConstantCurve`).
        pantoscopic_tilt: combiner forward tilt about world-x through the
            pupil (build-time frame, not a design variable).
        projector_tilt: beam-axis tilt about world-x from straight down.
        chassis_z: glass thickness; all thickness-dependent geometry is
            rederived from it.
        **geometry_kwargs: forwarded to
            :func:`apollo14.perseus.build_perseus_geometry` (mirror angle,
            chassis x/y, eye relief, pupil/aperture dims, light position).

    Returns:
        :class:`OpticalSystem` with chassis, aperture, mirrors, pupil.
    """
    if probe_wavelengths is None:
        probe_wavelengths = _DEFAULT_PROBE_WAVELENGTHS
    probe_wavelengths = jnp.asarray(probe_wavelengths)

    geometry = build_perseus_geometry(
        spacings=params.spacings,
        pantoscopic_tilt=pantoscopic_tilt, projector_tilt=projector_tilt,
        chassis_z=chassis_z, **geometry_kwargs)

    num_mirrors = params.curves.amplitude.shape[0]

    system = OpticalSystem(env_material=air)
    system.add(geometry.chassis)
    system.add(geometry.aperture)
    for mirror_idx in range(num_mirrors):
        system.add(PartialMirror(
            name=f"mirror_{mirror_idx}",
            position=geometry.mirror_positions[mirror_idx],
            normal=geometry.mirror_normal.copy(),
            width=geometry.mirror_width,
            height=geometry.mirror_height,
            wavelengths=probe_wavelengths,
            curve=params.curves.at(mirror_idx),
        ))
    system.add(geometry.pupil)
    return system
