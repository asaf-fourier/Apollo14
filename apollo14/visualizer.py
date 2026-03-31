import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

from apollo14.system import OpticalSystem
from apollo14.elements.surface import PartialMirror
from apollo14.elements.glass_block import GlassBlock
from apollo14.elements.aperture import RectangularAperture
from apollo14.elements.pupil import Pupil
from apollo14.tracer import TraceResult
from apollo14.geometry import compute_local_axes


def plot_system(system: OpticalSystem, trace_results: list[TraceResult] = None,
                figsize=(12, 10), elev=25, azim=-60):
    """Render the optical system in 3D with optional traced rays."""
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    for elem in system.elements:
        if isinstance(elem, GlassBlock):
            _draw_glass_block(ax, elem)
        elif isinstance(elem, PartialMirror):
            _draw_mirror(ax, elem)
        elif isinstance(elem, RectangularAperture):
            _draw_aperture(ax, elem)
        elif isinstance(elem, Pupil):
            _draw_pupil(ax, elem)

    if trace_results:
        for tr in trace_results:
            _draw_trace(ax, tr)

    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    ax.view_init(elev=elev, azim=azim)
    _set_equal_aspect(ax)
    plt.tight_layout()
    return fig, ax


def _draw_glass_block(ax, block: GlassBlock):
    for face in block.faces:
        verts = np.array(face.vertices)
        poly = Poly3DCollection([verts], alpha=0.08, facecolor='cyan', edgecolor='steelblue', linewidth=0.5)
        ax.add_collection3d(poly)


def _draw_mirror(ax, mirror: PartialMirror):
    pos = np.array(mirror.position)
    n = np.array(mirror.normal)
    lx, ly = compute_local_axes(mirror.normal)
    lx, ly = np.array(lx), np.array(ly)
    hw, hh = mirror.width / 2.0, mirror.height / 2.0

    corners = np.array([
        pos - hw * lx - hh * ly,
        pos + hw * lx - hh * ly,
        pos + hw * lx + hh * ly,
        pos - hw * lx + hh * ly,
    ])
    poly = Poly3DCollection([corners], alpha=0.3, facecolor='gold', edgecolor='darkorange', linewidth=1)
    ax.add_collection3d(poly)


def _draw_aperture(ax, aperture: RectangularAperture):
    pos = np.array(aperture.position)
    lx, ly = compute_local_axes(aperture.normal)
    lx, ly = np.array(lx), np.array(ly)
    hw, hh = aperture.width / 2.0, aperture.height / 2.0

    corners = np.array([
        pos - hw * lx - hh * ly,
        pos + hw * lx - hh * ly,
        pos + hw * lx + hh * ly,
        pos - hw * lx + hh * ly,
    ])
    poly = Poly3DCollection([corners], alpha=0.4, facecolor='black', edgecolor='gray', linewidth=1)
    ax.add_collection3d(poly)


def _draw_pupil(ax, pupil: Pupil):
    pos = np.array(pupil.position)
    lx, ly = compute_local_axes(pupil.normal)
    lx, ly = np.array(lx), np.array(ly)

    theta = np.linspace(0, 2 * np.pi, 40)
    circle = np.array([pos + pupil.radius * (np.cos(t) * lx + np.sin(t) * ly) for t in theta])
    poly = Poly3DCollection([circle], alpha=0.2, facecolor='green', edgecolor='darkgreen', linewidth=1)
    ax.add_collection3d(poly)


def _draw_trace(ax, trace: TraceResult):
    if not trace.hits:
        return

    points = [np.array(trace.hits[0].point)]
    for hit in trace.hits:
        points.append(np.array(hit.point))

    points = np.array(points)
    ax.plot(points[:, 0], points[:, 1], points[:, 2], 'r-', linewidth=0.5, alpha=0.6)


def _set_equal_aspect(ax):
    """Set equal aspect ratio for 3D plot."""
    limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
    center = limits.mean(axis=1)
    max_range = (limits[:, 1] - limits[:, 0]).max() / 2.0
    ax.set_xlim3d([center[0] - max_range, center[0] + max_range])
    ax.set_ylim3d([center[1] - max_range, center[1] + max_range])
    ax.set_zlim3d([center[2] - max_range, center[2] + max_range])
