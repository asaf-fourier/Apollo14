# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

## Commands

```bash
uv sync                            # Install dependencies (uses uv + pyproject.toml)
pytest                             # Run all tests
pytest tests/test_jax_tracer.py    # Run JAX tracer tests
```

## Project Overview

Apollo14 is a rewrite of Apollo13 — a simulation engine for designing **compound optical combiners** for AR glasses. The combiner is a stack of cascaded thin partial mirrors glued together, cut at ~45 deg, and encased in ophthalmic glass to fit inside a regular eyeglass frame. Each mirror reflects a fraction of projector light toward the eye while transmitting the rest (plus ambient light) to the next mirror.

### Goals

1. **Simulate the combiner** — non-sequential (detailed, full ray tree) and JAX-native (fast, differentiable) ray tracing.
2. **Output mirror reflectance curves** — per mirror, per incidence angle, across the visible spectrum. A separate project will design the actual thin-film coatings from these curves.
3. **Enable JAX-based optimization** — the JAX tracer (`jax_tracer.py`) is fully differentiable so gradients flow through the simulation. Optimization lives in the `helios` package (see below). Primary optimization variables: per-color mirror reflectances and distances between mirrors; later possibly thickness and aperture.

## Architecture

### Two tracers, different purposes

- **Non-sequential tracer** (`tracer.py`) — full ray tree with branching at every surface. Python-based, uses `isinstance` dispatch. For detailed analysis and visualization.
- **JAX tracer** (`jax_tracer.py`) — path-based, pure JAX, no Python control flow on array values. Optical paths are defined as data (`PathStep` sequences), making the tracer system-agnostic. The generic entry point is `trace_path`; combiner-specific paths are built by `build_combiner_paths`. Fully differentiable via `jax.grad`. Three combiner convenience functions:
  - `trace_ray` — single ray
  - `trace_batch` — N rays with N different directions (angular scan)
  - `trace_beam` — N rays with shared direction (projector beam, paths built once outside vmap)

### Key modules

- `combiner.py` — `CombinerConfig` dataclass and `build_system()` for constructing the optical system
- `jax_tracer.py` — path-based differentiable tracer with `PathStep`, `CombinerParams`, `trace_path`, `build_combiner_paths`, and `params_from_config()`
- `tracer.py` — non-sequential tracer returning `TraceResult` with `TraceHit` tree
- `interaction.py` — `Interaction` enum (REFLECTED, TRANSMITTED, ENTERING, EXITING, TIR, ABSORBED)
- `geometry.py` — JAX-native primitives: `reflect`, `snell_refract`, `ray_plane_intersection`, `normalize`
- `materials.py` — `Material` class with wavelength-dependent refractive index interpolation
- `projector.py` — `Projector` class and `scan_directions` for FOV scanning
- `visualizer.py` — Plotly 3D visualization with angular slider
- `elements/` — `PartialMirror`, `GlassBlock`, `Pupil`, `RectangularAperture`, `BoundaryPlane`

### Helios — optimization package (`helios/`)

Helios is the optimization layer. It imports from `apollo14` but **`apollo14` must never import from `helios`** — this keeps the simulation library independent so Helios can later be split into its own repository.

#### Key modules

- `merit.py` — D65 white-balance merit function sampling pupil positions and FOV angles; `MeritConfig`, `evaluate_merit`, `simulate_pupil_response`

#### Optimization targets

- Target: ~10% of projector light reaches a ~10x10 mm pupil (eyebox) uniformly across ~20x20 deg FOV, while maximizing ambient transparency.
- Merit function: takes a base target efficiency, tries to make the entire pupil D65-white-balanced.
- Mirror reflectances are per-color (R/G/B) for white-balance optimization.

### Principles

- **JAX everywhere** in the ray tracing math — arrays, not scalar dataclasses. Critical for differentiability.
- **Clean API** — the system definition and tracers should be usable as a library by the optimization project.
- **Plotly for visualization** — interactive 3D renders of the system with slider to step through scan angles.
- **Interaction types are enums** — use `Interaction.REFLECTED` etc., never raw strings.

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
