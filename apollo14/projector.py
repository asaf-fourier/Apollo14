import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import jax.numpy as jnp

from apollo14.geometry import normalize
from apollo14.ray import Ray
from apollo14.units import nm


def load_spectrum_csv(path, column: str = "W") -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Load a spectral curve from a CSV with a ``wavelength`` column (in nm)
    and one or more radiance columns. Returns ``(wavelengths, radiance)`` —
    wavelengths in internal length units (``nm`` scaled), radiance as read.
    """
    path = Path(path)
    with path.open() as f:
        reader = csv.DictReader(f)
        wls, rads = [], []
        for row in reader:
            wls.append(float(row["wavelength"]))
            rads.append(float(row[column]))
    return jnp.asarray(wls) * nm, jnp.asarray(rads)


@dataclass
class Projector:
    """A projector that emits a 2D grid of collimated rays.

    The intensity_map is a 2D array (ny, nx) with values in [0, 1] representing
    the relative intensity of each pixel. The grid spans the beam_width x beam_height
    area, with each pixel emitting a collimated ray in the given direction.

    For angular scanning, the direction can be rotated per scan step.
    """
    position: jnp.ndarray     # (3,) projector location
    direction: jnp.ndarray    # (3,) main emission direction (normalized)
    beam_width: float         # physical width of the beam cross-section
    beam_height: float        # physical height of the beam cross-section
    intensity_map: jnp.ndarray  # (ny, nx) pixel intensities in [0, 1]
    spectrum: Optional[Tuple[jnp.ndarray, jnp.ndarray]] = None
    """Optional ``(wavelengths, radiance)`` spectral emission curve.
    Units: wavelengths in internal length units, radiance in W/sr/m²/nm."""

    @classmethod
    def uniform(cls, position, direction, beam_width, beam_height,
                nx: int, ny: int, intensity: float = 1.0, spectrum=None):
        """Create a projector with uniform intensity across all pixels."""
        return cls(
            position=position,
            direction=normalize(direction),
            beam_width=beam_width,
            beam_height=beam_height,
            intensity_map=jnp.full((ny, nx), intensity),
            spectrum=spectrum,
        )

    @classmethod
    def from_csv(cls, path, position, direction, beam_width, beam_height,
                 nx: int, ny: int, column: str = "W", intensity: float = 1.0):
        """Build a projector whose spectrum comes from a measured CSV.

        The CSV is expected to have a ``wavelength`` column (in nm) and one
        or more radiance columns (e.g. ``R``/``G``/``B``/``W``). ``column``
        selects which one to load (default ``"W"`` — the white channel).
        """
        wls, rad = load_spectrum_csv(path, column=column)
        return cls(
            position=position,
            direction=normalize(direction),
            beam_width=beam_width,
            beam_height=beam_height,
            intensity_map=jnp.full((ny, nx), intensity),
            spectrum=(wls, rad),
        )

    @property
    def nx(self) -> int:
        return self.intensity_map.shape[1]

    @property
    def ny(self) -> int:
        return self.intensity_map.shape[0]

    def _compute_basis(self, direction=None):
        """Compute the beam cross-section basis vectors for a given direction."""
        d = normalize(direction if direction is not None else self.direction)
        # Build local frame: local_x across beam width, local_y across beam height
        ref = jnp.where(jnp.abs(d[2]) < 0.9, jnp.array([0.0, 0.0, 1.0]),
                        jnp.array([1.0, 0.0, 0.0]))
        local_x = normalize(jnp.cross(d, ref))
        local_y = normalize(jnp.cross(local_x, d))
        return local_x, local_y

    def generate_rays(self, direction=None, wavelength=None) -> Ray:
        """Generate a batched ``Ray`` for the beam cross-section.

        The returned ``Ray`` has ``pos`` shape ``(N, 3)``, a shared
        ``dir`` of shape ``(3,)`` (collimated beam), and ``intensity``
        of shape ``(N,)``.

        When ``wavelength`` is given and the projector has a ``spectrum``,
        per-pixel intensities are multiplied by the spectral radiance at
        that wavelength — so the same projector instance can emit into
        multiple wavelength bins without rebuilding.
        """
        d = normalize(direction if direction is not None else self.direction)
        local_x, local_y = self._compute_basis(d)

        xs = jnp.linspace(-self.beam_width / 2, self.beam_width / 2, self.nx)
        ys = jnp.linspace(-self.beam_height / 2, self.beam_height / 2, self.ny)
        gx, gy = jnp.meshgrid(xs, ys)  # (ny, nx)
        gx_flat = gx.ravel()
        gy_flat = gy.ravel()

        offsets = gx_flat[:, None] * local_x[None, :] + gy_flat[:, None] * local_y[None, :]
        origins = self.position[None, :] + offsets  # (N, 3)

        intensities = self.intensity_map.ravel()
        if wavelength is not None and self.spectrum is not None:
            spec_wls, spec_rad = self.spectrum
            intensities = intensities * jnp.interp(
                jnp.asarray(wavelength), spec_wls, spec_rad)

        return Ray(pos=origins, dir=d, intensity=intensities)


def scan_directions(base_direction, x_fov, y_fov, num_x, num_y):
    """Generate a grid of scan directions covering the given FOV.

    Returns:
        directions: (num_y, num_x, 3) array of normalized direction vectors
        angles: (num_y, num_x, 2) array of (angle_x, angle_y) in radians
    """
    d = normalize(base_direction)

    # Build rotation basis
    ref = jnp.where(jnp.abs(d[2]) < 0.9, jnp.array([0.0, 0.0, 1.0]),
                    jnp.array([1.0, 0.0, 0.0]))
    axis_x = normalize(jnp.cross(d, ref))       # rotation axis for x-scan
    axis_y = normalize(jnp.cross(axis_x, d))     # rotation axis for y-scan

    ax = jnp.linspace(-x_fov / 2, x_fov / 2, num_x) if num_x > 1 else jnp.array([0.0])
    ay = jnp.linspace(-y_fov / 2, y_fov / 2, num_y) if num_y > 1 else jnp.array([0.0])

    directions = []
    angles = []
    for iy in range(len(ay)):
        row_dirs = []
        row_angles = []
        for ix in range(len(ax)):
            # Rodrigues rotation: first around axis_y (vertical scan), then axis_x (horizontal)
            rotated = _rodrigues(d, axis_y, ay[iy])
            rotated = _rodrigues(rotated, axis_x, ax[ix])
            row_dirs.append(normalize(rotated))
            row_angles.append(jnp.array([ax[ix], ay[iy]]))
        directions.append(jnp.stack(row_dirs))
        angles.append(jnp.stack(row_angles))

    return jnp.stack(directions), jnp.stack(angles)


def _rodrigues(v, k, theta):
    """Rotate vector v around axis k by angle theta (Rodrigues' formula)."""
    cos_t = jnp.cos(theta)
    sin_t = jnp.sin(theta)
    return v * cos_t + jnp.cross(k, v) * sin_t + k * jnp.dot(k, v) * (1 - cos_t)
