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

    ``nx``/``ny`` set the ray grid dimensions — the beam is sampled on an
    ``nx × ny`` lattice spanning ``beam_width × beam_height``. All rays
    start at intensity 1.0 before spectral and angular scaling.

    Angular roll-off is modelled as a linear falloff in the projector's
    local x/y axes, specified as ``fraction per radian``. For example, a
    2% drop at 6° maps to ``falloff_x = 0.02 / (6 * deg)``.

    For angular scanning, the direction can be rotated per scan step.
    """
    position: jnp.ndarray     # (3,) projector location
    direction: jnp.ndarray    # (3,) main emission direction (normalized)
    beam_width: float         # physical width of the beam cross-section
    beam_height: float        # physical height of the beam cross-section
    nx: int                   # number of rays across the beam width
    ny: int                   # number of rays across the beam height
    falloff_x: float = 0.0    # linear angular falloff along local x (per rad)
    falloff_y: float = 0.0    # linear angular falloff along local y (per rad)
    spectrum: Optional[Tuple[jnp.ndarray, jnp.ndarray]] = None
    """Optional ``(wavelengths, radiance)`` spectral emission curve.
    Units: wavelengths in internal length units, radiance in W/sr/m²/nm."""

    @classmethod
    def uniform(cls, position, direction, beam_width, beam_height,
                nx: int, ny: int, spectrum=None,
                falloff_x: float = 0.0, falloff_y: float = 0.0):
        """Create a projector with uniform intensity across all pixels."""
        return cls(
            position=position,
            direction=normalize(direction),
            beam_width=beam_width,
            beam_height=beam_height,
            nx=nx, ny=ny,
            falloff_x=falloff_x, falloff_y=falloff_y,
            spectrum=spectrum,
        )

    @classmethod
    def from_csv(cls, path, position, direction, beam_width, beam_height,
                 nx: int, ny: int, column: str = "W",
                 falloff_x: float = 0.0, falloff_y: float = 0.0):
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
            nx=nx, ny=ny,
            falloff_x=falloff_x, falloff_y=falloff_y,
            spectrum=(wls, rad),
        )

    def _compute_basis(self, direction=None):
        """Compute the beam cross-section basis vectors for a given direction."""
        d = normalize(direction if direction is not None else self.direction)
        # Build local frame: local_x across beam width, local_y across beam height
        ref = jnp.where(jnp.abs(d[2]) < 0.9, jnp.array([0.0, 0.0, 1.0]),
                        jnp.array([1.0, 0.0, 0.0]))
        local_x = normalize(jnp.cross(d, ref))
        local_y = normalize(jnp.cross(local_x, d))
        return local_x, local_y

    def _angular_gain(self, direction) -> jnp.ndarray:
        """Scalar emission gain for a beam aimed along ``direction``.

        Models a real projector's angular roll-off: when the beam is steered
        off its nominal optical axis (``self.direction``), the emitter puts
        out less light. Returns a scalar in ``[0, 1]`` that multiplies every
        ray's intensity — so all rays in one beam share the same gain
        (consistent with a collimated beam where the "angle" is a property
        of the whole bundle, not individual rays).

        The angle is decomposed into two components by projecting
        ``direction`` onto the projector's base local x/y axes::

            ax = arcsin(direction · base_x)   # tilt around base_y
            ay = arcsin(direction · base_y)   # tilt around base_x

        A linear falloff is then applied independently on each axis::

            gain = (1 - falloff_x · |ax|) · (1 - falloff_y · |ay|)

        clipped to ``[0, 1]``. For ``direction == self.direction`` the dot
        products are zero, so ``gain == 1``. The clip on the dot products
        guards ``arcsin`` against tiny numerical overshoot past ±1.
        """
        base_x, base_y = self._compute_basis(self.direction)
        ax = jnp.arcsin(jnp.clip(jnp.dot(direction, base_x), -1.0, 1.0))
        ay = jnp.arcsin(jnp.clip(jnp.dot(direction, base_y), -1.0, 1.0))
        return jnp.clip(
            (1.0 - self.falloff_x * jnp.abs(ax))
            * (1.0 - self.falloff_y * jnp.abs(ay)),
            0.0, 1.0,
        )

    def generate_rays(self, direction=None, wavelength=None) -> Ray:
        """Generate a batched ``Ray`` for the beam cross-section.

        The returned ``Ray`` has ``pos`` shape ``(N, 3)``, a shared
        ``dir`` of shape ``(3,)`` (collimated beam), and ``intensity``
        of shape ``(N,)``.

        Intensity is ``spectrum(wavelength) × angular_gain(direction)``,
        where ``angular_gain`` is a linear falloff from the projector's
        base direction along its local x/y axes.
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

        intensities = jnp.full((self.nx * self.ny,), self._angular_gain(d))

        if wavelength is not None and self.spectrum is not None:
            spec_wls, spec_rad = self.spectrum
            intensities = intensities * jnp.interp(
                jnp.asarray(wavelength), spec_wls, spec_rad)

        return Ray(pos=origins, dir=d, intensity=intensities)


_PLAYNITRIDE_CSV = Path(__file__).parent / "data" / "projector" / "PlayNitride_(-1-3)_APL05prc.csv"


class PlayNitrideLed(Projector):
    """Projector whose spectrum is the measured PlayNitride micro-LED curve.

    ``color`` selects which channel column to read: ``"R"``, ``"G"``, ``"B"``,
    or ``"W"`` (white).
    """

    @classmethod
    def create(cls, position, direction, beam_width, beam_height,
               nx: int, ny: int, color: str,
               falloff_x: float = 0.0, falloff_y: float = 0.0):
        if color not in ("R", "G", "B", "W"):
            raise ValueError(f"color must be one of R/G/B/W, got {color!r}")
        wls, rad = load_spectrum_csv(_PLAYNITRIDE_CSV, column=color)
        return cls(
            position=position,
            direction=normalize(direction),
            beam_width=beam_width,
            beam_height=beam_height,
            nx=nx, ny=ny,
            falloff_x=falloff_x, falloff_y=falloff_y,
            spectrum=(wls, rad / rad.max()),
        )


class FovGrid:
    """FOV scan grid — iterable collection of directions with grid metadata.

    Wraps :func:`scan_directions` output so callers can iterate over
    flattened directions and later map results back to the 2-D FOV grid
    via :attr:`angles` and :attr:`grid_shape`.

    Usage::

        grid = FovGrid(projector.direction, x_fov, y_fov, num_x, num_y)
        for direction in grid:
            ray = projector.generate_rays(direction=direction)
            ...
        # result array of shape (len(grid),) can be reshaped:
        result_2d = result.reshape(grid.grid_shape)
    """

    def __init__(self, base_direction, x_fov, y_fov, num_x, num_y):
        directions_grid, angles_grid = _build_scan_grid(
            base_direction, x_fov, y_fov, num_x, num_y)
        self.directions_grid = directions_grid    # (num_y, num_x, 3)
        self.angles_grid = angles_grid            # (num_y, num_x, 2)
        self.num_x = num_x
        self.num_y = num_y
        self._flat_directions = directions_grid.reshape(-1, 3)

    @property
    def grid_shape(self):
        """(num_y, num_x) — shape for reshaping flat results back to 2-D."""
        return (self.num_y, self.num_x)

    @property
    def flat_directions(self):
        """(A, 3) flattened direction vectors."""
        return self._flat_directions

    @property
    def flat_angles(self):
        """(A, 2) flattened (angle_x, angle_y) in radians."""
        return self.angles_grid.reshape(-1, 2)

    def __len__(self):
        return self._flat_directions.shape[0]

    def __iter__(self):
        for idx in range(len(self)):
            yield self._flat_directions[idx]

    def __getitem__(self, idx):
        return self._flat_directions[idx]


def scan_directions(base_direction, x_fov, y_fov, num_x, num_y):
    """Generate a grid of scan directions covering the given FOV.

    Returns:
        directions: (num_y, num_x, 3) array of normalized direction vectors
        angles: (num_y, num_x, 2) array of (angle_x, angle_y) in radians

    .. note:: Prefer :class:`FovGrid` for new code — it wraps this
       function and provides iteration + grid metadata.
    """
    return _build_scan_grid(base_direction, x_fov, y_fov, num_x, num_y)


def _build_scan_grid(base_direction, x_fov, y_fov, num_x, num_y):
    d = normalize(base_direction)

    ref = jnp.where(jnp.abs(d[2]) < 0.9, jnp.array([0.0, 0.0, 1.0]),
                    jnp.array([1.0, 0.0, 0.0]))
    axis_x = normalize(jnp.cross(d, ref))
    axis_y = normalize(jnp.cross(axis_x, d))

    ax = jnp.linspace(-x_fov / 2, x_fov / 2, num_x) if num_x > 1 else jnp.array([0.0])
    ay = jnp.linspace(-y_fov / 2, y_fov / 2, num_y) if num_y > 1 else jnp.array([0.0])

    directions = []
    angles = []
    for iy in range(len(ay)):
        row_dirs = []
        row_angles = []
        for ix in range(len(ax)):
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
