from dataclasses import dataclass
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from apollo14.units import nm


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


_DATA_DIR = Path(__file__).parent / "data"

air = Air()
agc_m074 = Material.from_csv("agc_m074", _DATA_DIR / "agc_m074.csv", wavelength_units=nm)
