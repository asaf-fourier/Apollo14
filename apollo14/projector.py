from dataclasses import dataclass

import jax.numpy as jnp

from apollo14.geometry import normalize


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
    wavelength: float
    intensity_map: jnp.ndarray  # (ny, nx) pixel intensities in [0, 1]

    @classmethod
    def uniform(cls, position, direction, beam_width, beam_height, wavelength,
                nx: int, ny: int, intensity: float = 1.0):
        """Create a projector with uniform intensity across all pixels."""
        return cls(
            position=position,
            direction=normalize(direction),
            beam_width=beam_width,
            beam_height=beam_height,
            wavelength=wavelength,
            intensity_map=jnp.full((ny, nx), intensity),
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

    def generate_rays(self, direction=None):
        """Generate all rays as arrays.

        Args:
            direction: Override emission direction (for angular scanning).
                       If None, uses self.direction.

        Returns:
            origins: (N, 3) ray origins
            directions: (N, 3) ray directions (all identical for collimated beam)
            intensities: (N,) per-ray intensity from the intensity map
            pixel_indices: (N, 2) the (iy, ix) index of each ray's pixel
        """
        d = normalize(direction if direction is not None else self.direction)
        local_x, local_y = self._compute_basis(d)

        # Grid coordinates centered on the beam
        xs = jnp.linspace(-self.beam_width / 2, self.beam_width / 2, self.nx)
        ys = jnp.linspace(-self.beam_height / 2, self.beam_height / 2, self.ny)
        gx, gy = jnp.meshgrid(xs, ys)  # (ny, nx)

        # Flatten
        gx_flat = gx.ravel()
        gy_flat = gy.ravel()

        # Origins: position + offset in the beam plane
        offsets = gx_flat[:, None] * local_x[None, :] + gy_flat[:, None] * local_y[None, :]
        origins = self.position[None, :] + offsets  # (N, 3)

        n = origins.shape[0]
        directions = jnp.broadcast_to(d[None, :], (n, 3))
        intensities = self.intensity_map.ravel()

        # Pixel indices
        iy, ix = jnp.meshgrid(jnp.arange(self.ny), jnp.arange(self.nx), indexing='ij')
        pixel_indices = jnp.stack([iy.ravel(), ix.ravel()], axis=1)

        return origins, directions, intensities, pixel_indices


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
