import numpy as np
import itertools
from PyQt5 import QtWidgets
from otk.sdb import qt
from otk.sdb import *
from otk.rt2 import *

sdb_glsl = """
DID getSDB1(in vec4 x)
{
    return DID(sphereSDF(vec3(0., 0., 0.5), 0.7, x.xyz), 0);
}

DID getSDB2(in vec4 x)
{
    return DID(sphereSDF(vec3(0., 0., -0.5), 0.8, x.xyz), 1);
}

DID getSDB3(in vec4 x) {
    return DID(circleSDF(vec2(0.1, 0.), 0.2, x.xy), 2);
}

mat4 rotation(in vec3 v, in float theta) {
    float cs = cos(theta);
    float sn = sin(theta);

    return mat4(vec4(cs + v.x*v.x*(1. - cs), v.x*v.y*(1. - cs) + v.z*sn, v.x*v.z*(1. - cs) - v.y*sn, 0.),
                vec4(v.x*v.y*(1. - cs) - v.z*sn, cs + v.y*v.y*(1. - cs), v.y*v.z*(1. - cs) + v.x*sn, 0.),
                vec4(v.x*v.z*(1. - cs) + v.y*sn, v.y*v.z*(1. - cs) - v.x*sn, cs + v.z*v.z*(1. - cs), 0.),
                vec4(0., 0., 0., 1.));
}


DID getSDB0(in vec4 x)
{
    float theta = iTime;
    mat4 S = rotation(normalize(vec3(1., 1., 0.)), theta);
    x = S*x;
    float edge_width = 0.03;
    return edgeDifferenceSDB(edgeUnionSDB(getSDB1(x), getSDB2(x), edge_width, 3), getSDB3(x), edge_width, 3);
}
"""

# s1 = Sphere(0.7, (0, 0, 0.5))
# s2 = Sphere(0.8, (0, 0, -0.5))
# s3 = InfiniteCylinder(0.2, (0.1, 0))
# s0 = DifferenceOp(UnionOp((s1, s2)), s3)
# ids = add_ids(s0)
# sdb_glsl = gen_isdb_code(s0, ids, set())


#surface = UnionOp([make_spherical_singlet(-1, 1, 0.5, 0.5, (x, y, 0), 'square') for x, y in itertools.product((-1, 0, 1), repeat=2)])
#surface = UnionOp([make_spherical_singlet(-1, 1, 0.5, 0.5, (0.5, -0.5, 0), 'square'), make_spherical_singlet(-1, 1, 0.5, 0.5, (0.5, 0.5, 0), 'square')])
#surface = make_spherical_singlet_square_array(-1, 1, 0.5, (1, 1), (10, 10), (0, 0, 0))

surface0 = UnionOp([make_toroidal_singlet((0.5, -1), (-1, 0.3), 0.5, 2**(-0.5), (x, y, 0), 'square') for x, y in itertools.product((-1, 0, 1), repeat=2)])

#surface = make_spherical_singlet(1, -1, 0.5, 0.5, (0, 0, 0), 'square')
surface1 = Sphere(0.1)
surface = UnionOp((surface0, surface1))
ids = add_ids(surface)
sdb_glsl = gen_isdb_code(surface, ids, set())
print(sdb_glsl)

rays = [np.asarray([[x, y, -1], [x, y, 0], [0, 0, 1]]) for x, y in itertools.product(np.linspace(-0.5, 0.5, 5), repeat=2)]

app = QtWidgets.QApplication([])
w = qt.SphereTraceViewer(sdb_glsl)
w.display_widget.set_rays(rays)
w.show()
app.exec()
