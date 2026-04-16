"""Eyebox response computation for combiner optimization.

Traces R/G/B rays across eyebox sample points and FOV angles via the
JAX tracer, producing a per-cell, per-angle, per-wavelength intensity
tensor. Fully differentiable — gradients flow through intensity values.
"""

import jax.numpy as jnp

from apollo14.geometry import planar_grid_points
from apollo14.trace import trace_rays
from apollo14.binning import bin_hits_to_nearest

from helios.merit import DEFAULT_WAVELENGTHS


# ── Eyebox sampling ─────────────────────────────────────────────────────────

def eyebox_grid_points(center, normal, radius, nx, ny):
    """Dense ``(nx, ny)`` grid on the eyebox plane, spanning ±``radius``."""
    return planar_grid_points(center, normal, radius, radius, nx, ny)


# ── Response computation ────────────────────────────────────────────────────

def compute_eyebox_response(routes_per_wavelength, projector,
                            fov_grid, eyebox_points,
                            wavelengths=None):
    """Compute intensity at each eyebox sample for each FOV angle and wavelength.

    For each FOV direction in ``fov_grid``, traces a dense beam of rays
    from the projector through every pupil-terminated branch route,
    binning hits to the nearest eyebox grid point. Gradients flow
    through intensity values; spatial assignment is fixed
    (``stop_gradient`` on argmin).

    Args:
        routes_per_wavelength: ``(n_wavelengths, n_branches)`` list of
            pupil-terminated ``Route``s — e.g. from
            ``helios.merit.build_combiner_pupil_routes``.
        projector: :class:`Projector` used for all wavelengths. Defines
            position, direction, beam geometry, and spectrum.
        fov_grid: :class:`FovGrid` defining the FOV scan directions.
            The caller constructs this and can later use
            ``fov_grid.grid_shape`` and ``fov_grid.angles_grid`` to map
            the flat angle axis of the response back to 2-D FOV.
        eyebox_points: (S, 3) sample points on the eyebox
        wavelengths: ``(n_wavelengths,)`` trace wavelengths. Defaults to
            :data:`DEFAULT_WAVELENGTHS`.

    Returns:
        response: (S, A, n_wavelengths) intensity per sample, per angle,
            per wavelength — A matches ``len(fov_grid)``
    """
    if wavelengths is None:
        wavelengths = DEFAULT_WAVELENGTHS
    if len(wavelengths) != len(routes_per_wavelength):
        raise ValueError(
            f"wavelengths ({len(wavelengths)}) must match "
            f"routes_per_wavelength ({len(routes_per_wavelength)}) in length")

    wavelength_responses = []
    for wl_idx, branch_routes in enumerate(routes_per_wavelength):
        trace_wavelength = wavelengths[wl_idx]
        angle_responses = []
        for direction in fov_grid:
            ray = projector.generate_rays(direction=direction,
                                          wavelength=trace_wavelength)
            binned = jnp.zeros(eyebox_points.shape[0])
            for route in branch_routes:
                traced = trace_rays(route, ray, wavelength=trace_wavelength)
                binned = binned + bin_hits_to_nearest(
                    traced, eyebox_points, stop_grad=True)
            angle_responses.append(binned)

        per_wavelength = jnp.stack(angle_responses, axis=1)  # (S, A)
        wavelength_responses.append(per_wavelength)

    response = jnp.stack(wavelength_responses, axis=-1)  # (S, A, n_wl)
    return response
