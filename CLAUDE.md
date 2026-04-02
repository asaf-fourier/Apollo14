# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

## Commands

```bash
uv sync                            # Install dependencies (uses uv + pyproject.toml)
pytest                             # Run all tests
pytest tests/unit/                 # Run unit tests only
```

## Project Overview

Apollo14 is a rewrite of Apollo13 — a simulation engine for designing **compound optical combiners** for AR glasses. The combiner is a stack of cascaded thin partial mirrors glued together, cut at ~45 deg, and encased in ophthalmic glass to fit inside a regular eyeglass frame. Each mirror reflects a fraction of projector light toward the eye while transmitting the rest (plus ambient light) to the next mirror.

### Goals

1. **Simulate the combiner** — both sequential (fast, differentiable) and non-sequential (detailed reporting) ray tracing.
2. **Output mirror reflectance curves** — per mirror, per incidence angle, across the visible spectrum. A separate project will design the actual thin-film coatings from these curves.
3. **Enable JAX-based optimization** — the sequential tracer must be JAX-native so gradients flow through the simulation. Optimization itself lives in a separate project but Apollo14 must expose the right API. Primary optimization variables: distances between mirrors; later possibly thickness and aperture.

### Optical merit (for context — optimization lives externally)

- Target: ~10% of projector light reaches a ~10x10 mm pupil (eyebox) uniformly across ~20x20 deg FOV, while maximizing ambient transparency.
- Merit function (external project): takes a base target efficiency, tries to make the entire pupil D65-white-balanced.

## Architecture Principles

- **JAX everywhere** in the ray tracing math — arrays, not scalar dataclasses. This is critical for future differentiability.
- **Clean API** — the system definition and tracer should be usable as a library by the optimization project.
- **Plotly for visualization** — interactive 3D renders of the system with slider to step through scan angles.
- **Sequential tracer first** — fast path for optimization. Non-sequential tracer for detailed analysis after.

## Target System (Talos)

Reference implementation: Apollo13's `system/talos/talos.py`. Key parameters:

- 6 cascaded partial mirrors inside a glass chassis (~14x20x2 mm)
- Mirrors tilted at ~48 deg from horizontal, spaced ~1.47 mm apart
- Projector above the chassis, beam aimed downward
- Exit pupil at ~15 mm eye relief
- Glass substrate: AGC M-074 (or similar ophthalmic glass)
- Each mirror reflects a uniform fraction; the per-mirror reflectance compensates for upstream losses so equal light reaches the pupil from each mirror

## Heritage (Apollo13)

Apollo13 used numpy-based scalar ray tracing with custom Vector3D/Ray frozen dataclasses. Apollo14 replaces this with JAX arrays for differentiability and batched computation. Do not carry over Apollo13's dataclass-heavy patterns — prefer flat JAX arrays and functional style.

## Dependencies

- Python >= 3.14
- JAX (core computation)
- Plotly (visualization)
- NumPy (interop where needed)
