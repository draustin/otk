"""Definitions and operations for  homogeneous vectors in 3D with broadcasting.

Transformation matrices are applied on the right i.e. we use row vectors.

# History:

## 2018-07-13

Originally (pre July 13 2018), all points/vectors were 2D arrays, with coordinates running of the first dimension. This
is equivalent to a matrix in linear algebra. So we could think of all transformations as matrix multiplies. I choose this
for simplicity and speed. Matrix multiplication uses numpy.matmul. A second dimension can be appended if desired.
For speed check, see check_matrix_multiply.py. It is most efficient to store points and vectors with w included.

But it has several downsides:
    * Only one 'working' dimension for operations. To sample a 2D grid, need to flatten, perform operations, then reshape.
    * Not compatible with Numpy generalized ufuncs in which the core dimensions are spatial. This is because GUFuncs want
      core dimensions to be trailing.

So decided to change over to spatial coordinates in last dimension. Still using 4-vectors.

Transformations performed with numpy.matmul, which means transform matrix is on the right. So we have row, not column
vectors. For other dimensions, broadcasting is implied.
"""
import numpy as np
import numba

__all__ = ['dot', 'normalize', 'norm', 'cross', 'transform', 'stack_xyzw']

# TODO make numba optional
@numba.guvectorize([(numba.float64[:], numba.float64[:], numba.float64[:]),
                    (numba.complex128[:], numba.complex128[:], numba.complex128[:])], '(n),(n)->()', nopython=True)
def _dot(a, b, c):
    """c = sum_n[a[n]*conj(b[n]) i.e. second argument, b, is conjugated."""
    c[0] = 0
    for i in range(len(a)):
        c[0] += a[i]*np.conj(b[i])  # return np.einsum('ij, ij->j', v1, v2.conj())


def dot(a: np.ndarray, b: np.ndarray = None) -> np.ndarray:
    """

    Args:
        a: ...x4 array. Unconjugated argument of dot product.
        b: ...x4 array. Conjugated argument of dot product.

    Returns:
        ...x1 array: Dot product(s).
    """
    if b is None:
        b = a
    return _dot(a, b)[..., None]


def normalize(vector):
    with np.errstate(invalid='ignore'):
        c = vector/dot(vector)**0.5
    c[np.isnan(c)] = 0
    return c

def norm(vector):
    return dot(vector)**0.5


def to_xyzw(matrix):
    """Convenience/readibility function to bring spatial (trailing) axis to start.

    Args:
        matrix (...x4 array): Input matrix.

    Returns:
        4x... array
    """
    return np.rollaxis(matrix, -1)


def to_xyz(matrix):
    """Convenience/readibility function to pick x, y & z components and bring spatial (trailing) axis to start.

    Args:
        matrix (...x4 array): Input matrix.

    Returns:
        2x... array
    """
    return np.rollaxis(matrix[..., :3], -1)


def to_xy(matrix):
    """Convenience/readibility function to pick x & y components and bring spatial (trailing) axis to start.

    Args:
        matrix (...x4 array): Input matrix.

    Returns:
        2x... array
    """
    return np.rollaxis(matrix[..., :2], -1)


def transform(array, matrix):
    """Transform points/vectors by a matrix.

    Args:
        array (...x4 array): Input points/vector.
        matrix (4x4 array): Transformation matrix.

    Returns:
        ...x4 array: Transformed points/vectors.
    """
    return np.matmul(array, matrix)


def _cross(v1, v2):
    """Cross product.

    To be consistent with dot and triple product, result is conjugated.

    v1 = np.random.rand(4,4096)
    v2 = np.random.rand(4,4096)
    %timeit rt.math._cross(v1, v2) gives 83 us.

    Args:
        v1:
        v2:

    Returns:

    """
    result3 = np.cross(v1[:3, :], v2[:3, :], axis=0).conj()
    r = np.empty(np.broadcast(v1, v2).shape, result3.dtype)
    r[:3, :] = result3
    r[3, :] = 0
    return r


@numba.guvectorize([(numba.float64[:], numba.float64[:], numba.float64[:]),
                    (numba.complex128[:], numba.complex128[:], numba.complex128[:])], '(n),(n)->(n)', nopython=True)
def cross(a, b, c):
    """
    To be consistent with dot and triple product, result is conjugated.

    Args:
        a:
        b:
        c:

    Returns:

    """
    c[0] = np.conj(a[1]*b[2] - a[2]*b[1])
    c[1] = np.conj(a[2]*b[0] - a[0]*b[2])
    c[2] = np.conj(a[0]*b[1] - a[1]*b[0])
    c[3] = 0


# @numba.jit#(nopython=True)
# def cross(v1, v2):
#     """
#
#     v1 = np.random.rand(4,4096)
#     v2 = np.random.rand(4,4096)
#
#     %timeit rt.math.cross(v1, v2) returns 15.8 us on Dane's Laptop.
#
#     Args:
#         v1:
#         v2:
#
#     Returns:
#
#     """
#     assert v1.shape == v2.shape
#     length = v1.shape[1]
#     r = np.empty((4, length), type(v1[0, 0]*v2[0, 0]))
#     for j in range(length):
#         r[0, j] = np.conj(v1[1, j]*v2[2, j] - v1[2, j]*v2[1, j])
#         r[1, j] = np.conj(v1[2, j]*v2[0, j] - v1[0, j]*v2[2, j])
#         r[2, j] = np.conj(v1[0, j]*v2[1, j] - v1[1, j]*v2[0, j])
#         r[3, j] = 0
#     return r

@numba.guvectorize([(numba.float64[:], numba.float64[:], numba.float64[:], numba.float64[:]),
                    (numba.complex128[:], numba.complex128[:], numba.complex128[:], numba.complex128[:])],
                   '(n),(n),(n)->()', nopython=True)
def triple_(a, b, c, r):
    """Determinant of matrix formed by a, b and c vectors.

    No complex conjugation i.e. with our definitions, the result is
        dot(a, cross(b, c))
    since dot conjugates its second argument and cross conjugates the result. To see this, just look at the expression.

    Cyclic permutation of rows/columns of determinant also proves that the result is equal to dot(b, cross(c, a)) and
    dot(c, cross(a, b)).
    """
    r[0] = a[0]*(b[1]*c[2] - b[2]*c[1]) + a[1]*(b[2]*c[0] - b[0]*c[2]) + a[2]*(b[0]*c[1] - b[1]*c[0])


def triple(a, b, c):
    """a dot (b cross c)"""
    return triple_(a, b, c)[..., None]


# @numba.jit
# def triple(a, b, c):
#     """Scalar triple product, a dot (b cross c).
#
#     Args:
#         a, b, c (4xn array): Inputs.
#     """
#     # length = a.shape[1]
#     # r = np.empty(length, type(a[0, 0]*b[0, 0]*c[0, 0]))
#     # for j in range(length):
#     #     r[j] = (a[0, j]*(b[1, j]*c[2, j] - b[2, j]*c[1, j]) + a[1, j]*(b[2, j]*c[0, j] - b[0, j]*c[2, j]) +
#     #             a[2, j]*(b[0, j]*c[1, j] - b[1, j]*c[0, j]))
#     # return r
#     return dot(a, cross(b, c))

def stack_xyzw(x, y, z, w):
    """Stack Cartesian axes into a point or array of points.

    Broadcasts and stacks along new axis -1.

    Args:
        x, y, z: Coordinates to stack.

    Returns:
        ...x4 array: Stacked coordinates.
    """
    x, y, z, w = np.broadcast_arrays(x, y, z, w)
    point = np.stack((x, y, z, w), -1)
    return point


def concatenate_xyzw(x, y, z, w):
    """Concatenate Cartesian axes into a point or array of points.

    Broadcasts and concatenates along existing axis -1.

    Args:
        x, y, z: Coordinates to concatenate.

    Returns:
        ...x4 array: Concatenated coordinates.
    """
    x, y, z, w = np.broadcast_arrays(x, y, z, w)
    point = np.concatenate((x, y, z, w), -1)
    return point


def repr_transform(matrix):
    # return '(%s, %s, %s, %.0f), (%s, %s, %s, %.0f), (%s, %s, %s, %.0f),(%s, %s, %s, %.0f)'%tuple('%.3f'%e for e in matrix.ravel())
    return '[' + ', '.join('(' + ', '.join('%.3f'%(element*1e3) for element in row) + ')' for row in matrix) + '] mm'

def apply_bundle_fourier_transform(bundle0, n, f):
    xo0, yo0 = to_xy(bundle0.origin)
    xo1, yo1 = to_xy(bundle0.vector)/bundle0.vector[..., 2]*f
    origin1 = stack_xyzw(xo1, yo1, 0, 1)
    vector1 = normalize(stack_xyzw(-xo0, -yo0, f, 0))
    bundle1 = Line(origin1, vector1)
    # See Dane's logbook 2 p159.
    opl = -dot(bundle0.origin, bundle0.vector)*n
    return bundle1, opl