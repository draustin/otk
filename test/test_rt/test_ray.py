import otk.h4t
import otk.rt.lines
import numpy as np

import otk.rt.raytrace
from otk import rt, paraxial, ri

def test_ray():
    half_angle = 1e-3
    num_x = 7
    num_y = 9
    pol = [0, 1, 0, 0]
    ray = rt.Ray.make_filled_pyramid(half_angle, 800e-9, num_x, num_y, pol)
    assert np.allclose(ray.line.origin, [0, 0, 0, 1])
    assert ray.line.vector.shape == (num_x, num_y, 4)
    assert np.allclose(ray.line.vector[3, 4, :], [0, 0, 1, 0])
    assert np.allclose(ray.line.vector[-1, 4, :], [np.sin(half_angle), 0, np.cos(half_angle), 0])

def test_refraction():
    """Test collimation of a point source by a spherical refracting surface."""
    n1 = 1.5
    n2 =  3
    f = 100
    r0 = np.asarray((100, 20, 300, 1))
    interface = rt.FresnelInterface(ri.FixedIndex(n1), ri.FixedIndex(n2))
    s = rt.Surface(rt.SphericalProfile(f*(n2 - n1)), otk.h4t.make_translation(*r0[:3]), interface=interface)
    origin = r0 + (0, 0, -f*n1, 0)
    line = rt.Line(origin, rt.normalize((0.001, 0.001, 1, 0)))
    pol = rt.cross(line.vector, [0,1,0,0])
    ray = rt.raytrace.Ray(line, pol, 0, 860e-9, n1)
    segments = ray.trace_surfaces((s,), ('transmitted', ))[0]
    assert len(segments) == 2
    assert segments[-1].ray.n == n2
    assert np.allclose(segments[-1].ray.line.vector, (0, 0, 1, 0))

    # Old optical graph test code. Leave here for when we rebuild optical graph.
    # nodes, edges = og.trace_surface_path((s,), ('transmitted',), ray)
    # assert len(nodes) == 5 # Initial, intersection, boundary, mask, transmitted.
    # assert len(edges) == len(nodes)-1
    # assert nodes[-1].field.n == n2


def test_make_spherical_lens_surfaces():
    roc1 = 100
    roc2 = 50
    d = 30
    n = 1.5
    f, h1, h2 = paraxial.calc_thick_spherical_lens(n, roc1, -roc2, d)
    surfaces = rt.make_spherical_lens_surfaces(roc1, -roc2, d, ri.FixedIndex(n))
    line = rt.Line((0, 0, -f + h1 - d/2, 1), rt.normalize((1e-4, 1e-4, 1, 0)))
    pol = rt.cross(line.vector, [0,1,0,0])
    ray = rt.raytrace.Ray(line, pol, 0, 860e-9, 1)
    segments = ray.trace_surfaces(surfaces, ['transmitted']*2)[0]
    assert len(segments) == 3
    assert segments[1].ray.n == n
    assert np.allclose(segments[-1].ray.line.vector, (0, 0, 1, 0))

    # Old optical graph test code. Leave here for when we rebuild optical graph.
    # nodes, edges = og.trace_surface_path(surfaces, ['transmitted']*2, ray)
    # assert len(nodes) == 9 # Initial, intersection, boundary, mask, refracted, intersection, boundary, mask, refracted.
    # assert nodes[4].field.n == n
    # assert np.allclose(nodes[-1].field.line.vector, (0, 0, 1, 0))

def test_reflect_Surface():
    s = rt.Surface(rt.PlanarProfile(), interface=rt.Mirror())
    s.rotate_y(np.pi/4)
    line = rt.Line((0, 0, -1, 1), (0, 0, 1, 0))
    pol = rt.cross(line.vector, [0,1,0,0])
    ray = rt.raytrace.Ray(line, pol, 0, 860e-9, 1)
    segments = ray.trace_surfaces([s], ['reflected'])[0]
    assert len(segments) == 2
    assert np.allclose(segments[-1].ray.line.vector, (-1, 0, 0, 0))

    # Old optical graph test code. Leave here for when we rebuild optical graph.
    # nodes, edges = og.trace_surface_path([s], ['reflected'], ray)
    # assert np.allclose(nodes[-1].field.


