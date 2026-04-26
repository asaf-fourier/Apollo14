from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from apollo14.units import nm

# Common visible-spectrum grid. All Material.data pytrees are resampled onto
# this grid so every route surface has identically-shaped n1/n2 leaves
# (required for stacking into a single lax.scan pytree).
STANDARD_WAVELENGTHS = jnp.linspace(380.0 * nm, 780.0 * nm, 81, dtype=jnp.float32)


class MaterialData(NamedTuple):
    """Pytree-friendly sampled material: n(wavelength) via linear interp."""
    wavelengths: jnp.ndarray  # (K,)
    n_values: jnp.ndarray     # (K,)


@dataclass(frozen=True)
class Material:
    name: str
    wavelengths: jnp.ndarray  # (N,) in internal units (mm)
    n_values: jnp.ndarray     # (N,) real refractive index
    k_values: jnp.ndarray     # (N,) extinction coefficient

    def n(self, wavelength):
        return jnp.interp(wavelength, self.wavelengths, self.n_values)

    def k(self, wavelength):
        return jnp.interp(wavelength, self.wavelengths, self.k_values)

    @property
    def data(self) -> MaterialData:
        """Pytree-friendly sampled (wavelength, n) pair on the standard grid.

        All materials share ``STANDARD_WAVELENGTHS`` so route surfaces can
        be stacked into a single pytree with uniform leaf shapes.
        """
        n = jnp.interp(STANDARD_WAVELENGTHS,
                       jnp.asarray(self.wavelengths, dtype=jnp.float32),
                       jnp.asarray(self.n_values, dtype=jnp.float32))
        return MaterialData(wavelengths=STANDARD_WAVELENGTHS, n_values=n)

    @classmethod
    def from_csv(cls, name: str, path: str, wavelength_units: float = nm):
        data = np.loadtxt(path)
        wavelengths = data[:, 0] * wavelength_units
        n_values = data[:, 1]
        k_values = data[:, 2] if data.shape[1] > 2 else np.zeros_like(n_values)
        return cls(
            name=name,
            wavelengths=jnp.array(wavelengths),
            n_values=jnp.array(n_values),
            k_values=jnp.array(k_values),
        )


@dataclass(frozen=True)
class Air(Material):
    name: str = "air"
    wavelengths: jnp.ndarray = None
    n_values: jnp.ndarray = None
    k_values: jnp.ndarray = None

    def n(self, wavelength):
        return 1.0

    def k(self, wavelength):
        return 0.0

    @property
    def data(self) -> MaterialData:
        # Constant n=1 across the standard grid.
        return MaterialData(
            wavelengths=STANDARD_WAVELENGTHS,
            n_values=jnp.ones_like(STANDARD_WAVELENGTHS),
        )


_DATA_DIR = Path(__file__).parent / "data"

air = Air()
agc_m074 = Material.from_csv("agc_m074", _DATA_DIR / "agc_m074.csv", wavelength_units=nm)
