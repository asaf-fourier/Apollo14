# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

## Commands

```bash
uv sync                            # Install dependencies (uses uv + pyproject.toml)
pytest                             # Run all tests
pytest tests/test_jax_tracer.py    # Run JAX tracer tests
```

## Project Overview

Apollo14 is a **JAX-native sequential ray tracer** for designing compound optical combiners for AR glasses. The combiner is a stack of cascaded thin partial mirrors glued together, cut at ~45 deg, and encased in ophthalmic glass to fit inside a regular eyeglass frame. Each mirror reflects a fraction of projector light toward the eye while transmitting the rest (plus ambient light) to the next mirror.

The tracer is fully differentiable — `jax.grad` flows through the entire simulation, enabling gradient-based optimization of mirror reflectances, spacing, and other parameters.

### Goals

1. **Simulate the combiner** — JAX-native sequential ray tracing: fast, vectorized (`vmap`), JIT-compiled, and differentiable.
2. **Output mirror reflectance curves** — per mirror, per incidence angle, across the visible spectrum. A separate project will design the actual thin-film coatings from these curves.
3. **Enable JAX-based optimization** — the tracer is fully differentiable so gradients flow through the simulation. Optimization lives in the `helios` package (see below). Primary optimization variables: per-color mirror reflectances and distances between mirrors; later possibly thickness and aperture.

## Architecture

### JAX sequential tracer

The tracer follows predefined optical paths (sequences of surfaces) rather than branching ray trees. Each optical element provides a `jax_interact()` method — a pure JAX function that computes intersections, refractions, or reflections without Python control flow on traced values. The tracer orchestrates these via `jax.lax.scan` over homogeneous sequences (e.g., the mirror stack), keeping dynamic path lengths while avoiding `jnp.where`-based type dispatch.

Key design:
- **Per-element JAX methods** — physics lives on the element (e.g., `PartialMirror.jax_interact()`), not in a monolithic tracer function.
- **Homogeneous scan** — main path is a `lax.scan` over mirrors (all same type), branch paths (exit face + pupil) are short and unrolled.
- **vmap for batching** — `trace_beam` vmaps over ray origins with shared direction; `trace_batch` vmaps over both origins and directions.
- **Differentiable** — `jax.grad` through the full pipeline for optimization.

### Key modules

- `combiner.py` — `build_default_system()` for constructing the Talos reference optical system
- `jax_tracer.py` — sequential differentiable tracer with `CombinerPath`, `trace_combiner_beam`, `trace_combiner_batch`, `trace_combiner_ray`
- `geometry.py` — JAX-native primitives: `reflect`, `snell_refract`, `ray_plane_intersection`, `normalize`
- `materials.py` — `Material` class with wavelength-dependent refractive index interpolation
- `projector.py` — `Projector` class and `scan_directions` for FOV scanning
- `visualizer.py` — Plotly 3D visualization with angular slider
- `elements/` — `PartialMirror`, `GlassBlock`, `Pupil`, `RectangularAperture`, `BoundaryPlane`

### Helios — optimization package (`helios/`)

Helios is the optimization layer. It imports from `apollo14` but **`apollo14` must never import from `helios`** — this keeps the simulation library independent so Helios can later be split into its own repository.

#### Key modules

- `merit.py` — D65 white-balance merit function; `MeritConfig`, `evaluate_merit`
- `eyebox.py` — eye-box area and uniformity merit functions; `EyeboxConfig`, `EyeboxAreaConfig`, `compute_eyebox_response`, `eyebox_merit`, `eyebox_area_merit`

#### Optimization targets

- Target: ~10% of projector light reaches a ~10x10 mm pupil (eyebox) uniformly across ~20x20 deg FOV, while maximizing ambient transparency.
- Merit function: takes a base target efficiency, tries to make the entire pupil D65-white-balanced.
- Mirror reflectances are per-color (R/G/B) for white-balance optimization.
- Eye-box area merit: maximize the number of pupil grid cells where all FOV angles exceed a minimum intensity threshold.

### Principles

- **JAX everywhere** in the ray tracing math — arrays, not scalar dataclasses. Critical for differentiability.
- **Physics on elements** — each optical element owns its JAX interaction method. The tracer orchestrates, elements compute.
- **Clean API** — the system definition and tracer should be usable as a library by the optimization project.
- **Plotly for visualization** — interactive 3D renders of the system with slider to step through scan angles.

## Naming conventions

Use **descriptive, self-explanatory names** for all variables. Avoid single-letter or abbreviated names — code should read clearly without comments explaining what a variable holds.

- `input_flux` not `I_in`, `mean_brightness` not `I_bar`, `brightness_threshold` not `I_thresh`
- `above_threshold` not `above`, `cell_mask` not `mask`, `num_target_cells` not `n_cells`
- `loss_coverage` not `L_cov`, `loss_shape` not `L_shape`
- `wavelength_offset` not `delta`, `exponent` not `expo`
- `moment` / `variance` / `step_count` not `m` / `v` / `t` (Adam state)
- `corrected_moment` not `m_hat`, `learning_rate` not `lr`
- `mirror_idx` not `i`, `mirror_amps` not `a`, `spacing` not `s` (loop variables)
- `clipped_spacings` not `sp`, `total_spacing` not `total`

Exceptions: standard mathematical notation in docstrings and comments is fine (e.g. `Ī(s)`, `σ`, `β` in formulas). The rule applies to Python identifiers, not documentation.

## Target System (Talos)

Reference implementation: Apollo13's `system/talos/talos.py`. Key parameters:

- 6 cascaded partial mirrors inside a glass chassis (~14x20x2 mm)
- Mirrors tilted at ~48 deg from horizontal, spaced ~1.47 mm apart
- Projector above the chassis, beam aimed downward
- Exit pupil at ~15 mm eye relief
- Glass substrate: AGC M-074 (or similar ophthalmic glass)
- Each mirror reflects a uniform fraction (5%); per-mirror reflectance ratio compensates for upstream losses so equal absolute intensity reaches the pupil from each mirror

## Heritage (Apollo13)

Apollo13 used numpy-based scalar ray tracing with custom Vector3D/Ray frozen dataclasses. Apollo14 replaces this with JAX arrays for differentiability and batched computation. Do not carry over Apollo13's dataclass-heavy patterns — prefer flat JAX arrays and functional style.

## Dependencies

- Python >= 3.14
- JAX (core computation)
- Plotly (visualization)
- NumPy (interop where needed)
