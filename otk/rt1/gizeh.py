from typing import Sequence
import numpy as np
import gizeh
from .. types import Matrix4, Sequence3
from . import Ray, RaySegment


def make_gizeh_wavefront(ray: Ray, phase: float, inverse_matrix: np.ndarray, color=(1, 0, 0)) -> gizeh.Element:
    """Make gizeh polyline representing a wavefront.

    Advances origin by phase and plots x, y components.
    """
    d = phase / ray.k
    points = np.matmul(ray.line.advance(d).origin, inverse_matrix)[..., :2].reshape((-1, 2))
    return gizeh.polyline(points, stroke=color, stroke_width=1)


def make_gizeh_ray(seg: RaySegment, inverse_matrix: Matrix4, stroke_width: float = 1) -> gizeh.Group:
    """Make Gizeh element containing a line for each ray in given segment.

    The transfrom from segment parent coordinate system to Gizeh element coordinates is given by inverse_matrix.
    """
    assert seg.length is not None  # todo more elegant way of dealing with open segments
    starts = np.matmul(seg.ray.line.origin, inverse_matrix)[..., :2].reshape((-1, 2))
    stops = np.matmul(seg.ray.line.advance(seg.length).origin, inverse_matrix)[..., :2].reshape((-1, 2))
    elements = []
    for start, stop in zip(starts, stops):
        elements.append(gizeh.polyline(np.vstack((start, stop)), stroke_width=stroke_width, stroke=(1, 0, 0)))
    return gizeh.Group(elements)


def make_multi_gizeh_ray(segments: Sequence[RaySegment], inverse_matrix: Matrix4, stroke_width: float = 1):
    liness = RaySegment.make_lines(segments)
    elements = []
    for lines in liness:
        lines_projected = np.matmul(lines, inverse_matrix)[..., :2]
        for line in lines_projected:
            element = gizeh.polyline(line, stroke_width=stroke_width, stroke=(1, 0, 0))
            elements.append(element)
    group = gizeh.Group(elements)
    return group

def make_multi_gizeh_wavefront(segments: Sequence[RaySegment], inverse_matrix: Matrix4, phase, color: Sequence3 = (1, 0, 0),
                               stroke_width: float = 2) -> gizeh.Element:
    """Draw wavefront in first segment that contains phase."""
    points_parent = RaySegment.connect_iso_phase(segments, phase)
    points = np.matmul(points_parent, inverse_matrix)[..., :2].reshape((-1, 2))
    element = gizeh.polyline(points, stroke_width=stroke_width, stroke=color)
    return element
