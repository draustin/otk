import numpy as np
import gizeh
from ..types import Matrix4, Sequence3
from .. import v4hb
from . import Segment


def make_gizeh(seg: Segment, matrix: Matrix4, fill:Sequence3=(1, 0, 0), num_points:int=16) -> gizeh.Element:
    """Make gizeh element.

    The transformation from coordinate system of segment to 2D scene is given by matrix.
    """
    assert all(surface is not None for surface in seg.surfaces), 'Segment must be bounded by surfaces'
    assert np.prod(seg.beam.mode.shape) == 1, 'Only one PBG allowed (for now)'
    length = np.asscalar(seg.length)  # Extra dimensions confuse things.
    origin = seg.beam.mode.line_base.origin.reshape((4,))
    vector = seg.beam.mode.line_base.vector.reshape((4,))
    rhat = v4hb.cross(matrix[2, :], vector)

    def calc_point(d, sign):
        """Calculate beam edge point.

        Args:
            d (scalar): Distance along beam.
            sign (scalar): Direction along rhat.

        Returns:
            ...x4 array: Beam edge point(s).
        """
        r = seg.beam.calc_edge_distance(d, sign * rhat)
        point = origin + d * vector + sign * rhat * r
        return point

    dss = []
    for sign in (-1, 1):
        dss.append([])
        for (surface, d_guess) in zip(seg.surfaces, (0, length)):
            d_root = surface.intersect_global_curve(lambda d: calc_point(d, sign).reshape((4,)), d_guess)
            dss[-1].append(d_root)

    dm = np.linspace(dss[0][0], dss[0][1], num_points)[:, None]
    pointsm = calc_point(dm, -1).reshape((-1, 4))
    dp = np.linspace(dss[1][1], dss[1][0], num_points)[:, None]
    pointsp = calc_point(dp, 1).reshape((-1, 4))
    to_section = np.linalg.inv(matrix)[:, :2]
    points_section = np.concatenate((pointsm, pointsp)).dot(to_section)
    return gizeh.polyline(points_section, close_path=True, fill=fill)