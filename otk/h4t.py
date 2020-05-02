"""Homogeneous 4x4 transformation matrices.

Spatial vectors are rows.
"""
import numpy as np
from .functions import normalize
from scipy.spatial.transform import Rotation
from .types import Sequence3, Matrix4

__all__ = ['make_x_rotation', 'make_y_rotation', 'make_z_rotation', 'make_translation', 'make_rotation',
    'make_scaling', 'make_frame', 'decompose_matrix', 'compose_matrix']


def make_x_rotation(theta: float, y0: float = 0., z0: float = 0.):
    """Right-hand convention - thumb points along x, fingers curl in positive theta direction.

    So with theta = 90, y*R = z.
    """
    cs = np.cos(theta)
    sn = np.sin(theta)
    rot = np.asarray([[1., 0., 0., 0.], [0., cs, sn, 0.], [0., -sn, cs, 0.], [0., 0., 0., 1.]])
    tr = make_translation(0., y0, z0)
    inv_tr = make_translation(0., -y0, -z0)
    return inv_tr.dot(rot).dot(tr)


def make_y_rotation(theta: float, x0=0, z0=0):
    """Right-hand convention - thumb points along y,  fingers curl in positive theta direction.
    So with theta = 90, z*R = x.
    """
    cs = np.cos(theta)
    sn = np.sin(theta)
    rot = np.asarray([[cs, 0., -sn, 0.], [0., 1., 0., 0.], [sn, 0., cs, 0.], [0., 0., 0., 1.]])
    tr = make_translation(x0, 0., z0)
    inv_tr = make_translation(-x0, 0., -z0)
    return inv_tr.dot(rot).dot(tr)


def make_z_rotation(theta: float, x0: float = 0, y0: float = 0):
    """With theta=90 deg, x*R = y."""
    cs = np.cos(theta)
    sn = np.sin(theta)
    rot = np.asarray([[cs, sn, 0., 0.], [-sn, cs, 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]])
    tr = make_translation(x0, y0, 0.)
    inv_tr = make_translation(-x0, -y0, 0.)
    return inv_tr.dot(rot).dot(tr)


def make_translation(x: float, y: float, z: float):
    return np.asarray([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [x, y, z, 1.]])


def make_rotation(vector: Sequence3, theta: float):
    """Make matrix representing rotation about arbitrary axis.

    Per package convention, matrix is applied on the right of row vectors.

    Right-hand convention i.e. with thumb along vector, positive theta is direction of finger curl.

    Args:
        vector: Axis of rotation. Must be normalized!
        theta: Rotation angle.

    Returns:
        4x4 array
    """
    cs = np.cos(theta)
    sn = np.sin(theta)
    x, y, z = vector[:3]
    # Source: https://en.wikipedia.org/wiki/Rotation_matrix.
    m = np.asarray([[cs + x**2*(1 - cs), x*y*(1 - cs) + z*sn, x*z*(1 - cs) - y*sn, 0],
                    [y*x*(1 - cs) - z*sn, cs + y**2*(1 - cs), y*z*(1 - cs) + x*sn, 0],
                    [z*x*(1 - cs) + y*sn, z*y*(1 - cs) - x*sn, cs + z**2*(1 - cs), 0], [0, 0, 0, 1]])
    return m


def make_scaling(x: float, y: float = None, z: float = None) -> Matrix4:
    if y is None:
        y = x
    if z is None:
        z = y
    return np.diag([x, y, z, 1.])


def make_frame(z: Sequence3, up: Sequence3 = None, right: Sequence3 = None, origin: Sequence3 = (0., 0., 0.)) -> Matrix4:
    """Make local to parent orthogonal matrix with given axes (in parent coordinates).

    One of up or right must be given - they become the y or x axes respectively after making orthogonal with z.
    """
    z = np.asarray(z, float)
    assert z.shape == (3,)

    origin = np.asarray(origin, float)
    assert origin.shape == (3,)

    z = normalize(z)
    if up is not None:
        up = np.asarray(up, float)
        assert up.shape == (3,)

        x = normalize(np.cross(up, z))
        y = np.cross(z, x)
    else:
        right = np.asarray(right, float)
        assert right.shape == (3,)

        y = normalize(np.cross(z, right))
        x = np.cross(y, z)

    return np.c_[np.stack((x, y, z, [origin[0], origin[1], origin[2]]), 0), [[0.], [0.], [0.], [1.]]]


def decompose_matrix(m, euler_seq:str = 'xyz'):
    """Convert 4x4 transformation matrix to 6-tuple of position, Euler angles."""
    r = tuple(m[3, :3])
    theta = tuple(Rotation.from_dcm(m[:3, :3]).to_euler(euler_seq))
    return r + theta


def compose_matrix(x, y, z, angle1, angle2, angle3, euler_seq:str='xyz'):
    """Convert 6-tuple of position & Euler angles to 4x4 transform matrix."""
    return np.c_[np.Rotation.from_euler(euler_seq, (angle1, angle2, angle3)).to_dcm(), [[0], [0], [0]], np.r_[x, y, z, 1]]