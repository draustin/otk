import numpy as np
from otk import ri
from otk import math as omath
from otk import paraxial
from otk.sdb import *
from otk.sdb.scalar import *
from otk.rt2 import *
from otk.rt2.scalar import *
from otk.sdb.lens import *
from otk import v4

def test_tracing():
    normal = 0, 0, -1
    constant = 0
    surface = Plane(normal, constant)
    deflector = make_fresnel_deflector()
    n0 = 1
    n0fun = ri.FixedIndex(n0)
    n1  = 1.5
    n1fun = ri.FixedIndex(n1)
    element = SimpleElement(surface, UniformIsotropic(n1fun), deflector)

    assert get_deflector(element, np.asarray((1, 2, 3, 1))) is deflector

    assembly = Assembly(element.surface, [element], UniformIsotropic(n0fun))
    assert assembly.get_transformed_element((1, 2, -1, 1)) is None
    assert assembly.get_transformed_element((1, 2, 1, 1)).element is element

    lamb = 800e-9
    incident_ray = make_ray(1, 2, -1, 0, 1, 1, 1, 0, 0, n0fun(lamb), 1, 0, lamb)
    epsilon = 1e-9
    length, deflected_rays = assembly.process_ray(incident_ray, dict(epsilon=epsilon, t_max=1e9, max_steps=100))
    assert (2**0.5 - epsilon) <= length  <= (2**0.5 + epsilon)

    (rp, rs), (tp, ts) = omath.calc_fresnel_coefficients(n0, n1, abs(v4.dot(incident_ray.line.vector, normal)))
    assert 0 <= getsdb(surface, deflected_rays[0].line.origin) <= epsilon
    assert np.array_equal(deflected_rays[0].line.vector, v4.normalize((0, 1, -1, 0)))
    assert np.isclose(deflected_rays[0].flux, rs**2)

    vy = 2**-0.5*n0/n1
    assert 0 >= getsdb(surface, deflected_rays[1].line.origin) >= -epsilon
    assert np.allclose(deflected_rays[1].line.vector, (0, vy, (1 - vy**2)**0.5, 0))
    assert np.isclose(deflected_rays[1].flux, ts**2*n1/n0)
    #assert np.array_equal(deflected_rays[0].vector, (0, 0, -1, 0))

def test_parabolic_mirror():
    roc = 0.1
    f = roc/2
    vertex = np.asarray((0, 0, 0))
    # Creater a parabolic mirror pointing along +z axis.
    surface = BoundedParaboloid(roc, 1.5*roc, True, vertex)
    deflector = make_constant_deflector(1, 1, 0, 0, True, False)
    n = ri.vacuum
    assembly = Assembly(surface, [SimpleElement(surface, UniformIsotropic(n), deflector)], UniformIsotropic(n))
    lamb = 800e-9
    epsilon = 1e-9
    focus = vertex + (0, 0, f)

    incident_ray = make_ray(*focus, 1, 1, 0, 1, -1, 0, n(lamb), 1, 0, lamb)
    length, deflected_rays = assembly.process_ray(incident_ray, dict(epsilon=epsilon, t_max=1e9, max_steps=100))
    assert len(deflected_rays) == 1
    assert np.allclose(deflected_rays[0].line.vector, (0, 0, 1, 0))

def test_biconvex_lens():
    ne = ri.air
    n = ri.fused_silica
    f = 0.1
    lamb = 800e-9
    thickness = 5e-3
    vertex0 = np.asarray((1, 2, 3, 1))
    vertex1 = vertex0 + (0, 0, thickness, 0)
    roc0, roc1 = paraxial.design_singlet(n(lamb), f, 0, thickness, ne(lamb))
    f_, h0, h1 = paraxial.calc_thick_spherical_lens(n(lamb), roc0, roc1, thickness, ne(lamb))
    assert np.isclose(f_, f)

    r = 20e-3
    surface = make_spherical_singlet(roc0, roc1, thickness, vertex0[:3], 'circle', r)
    # Vertices are on surface.
    assert np.isclose(getsdb(surface, vertex0), 0)
    assert np.isclose(getsdb(surface, vertex1), 0)
    assert np.isclose(getsdb(surface, vertex0 + (0, 0, -thickness, 0)), thickness)


    element = SimpleElement(surface, UniformIsotropic(n), perfect_refractor)
    assembly = Assembly(surface, [element], UniformIsotropic(ne))
    sphere_trace_kwargs = dict(epsilon=1e-9, t_max=1e9, max_steps=100)

    # Compute object and image points for 2f-2f imaging. zp is relative to vertex.
    zp_object = -(2*f - h0)
    zp_image = 2*f + h1 + thickness
    object = vertex0 + (0, 0, zp_object, 0)
    image = vertex0 + (0, 0, zp_image, 0)

    # Trace on-axis ray from object point.
    incident_ray0 = make_ray(*object[:3], 0, 0, 1, 0, 1, 0, ne(lamb), 1, 0, lamb)
    branch = assembly.nonseq_trace(incident_ray0, sphere_trace_kwargs)
    segments = branch.flatten()
    assert len(segments) == 3
    # Check distance to intersection.
    assert np.isclose(segments[0].length, -zp_object)
    # Check refracted ray.
    assert np.array_equal(segments[1].ray.line.vector, (0, 0, 1, 0))
    assert np.array_equal(segments[1].ray.k, n(lamb)*segments[1].ray.line.vector*2*np.pi/lamb)
    # Check distance in lens.
    assert np.isclose(segments[1].length, thickness)
    # Check output ray.
    assert np.allclose(segments[2].ray.line.origin, vertex1)
    assert np.array_equal(segments[2].ray.line.vector, (0, 0, 1, 0))
    assert np.array_equal(segments[2].ray.k, ne(lamb)*segments[2].ray.line.vector*2*np.pi/lamb)

    # Trace ray at angle.
    incident_ray1 = make_ray(*object[:3], 0.006, 0, 1, 0, 1, 0, ne(lamb), 1, 0, lamb)
    segments = assembly.nonseq_trace(incident_ray1, sphere_trace_kwargs).flatten()
    assert len(segments) == 3
    # Check ray passes close to image.
    assert v4.norm(segments[2].ray.line.pass_point(image)[1].origin - image) < 1e-6

    # TODO check phase

def test_hyperbolic_lens():
    # Prove that with a conic surface we can focus parallel rays perfectly.
    n1 = 1
    n2 = 1.5
    f = 1
    roc = f*(n2 - n1)/n2
    x0s = [-0.2, -0.1, 0, 0.1, 0.2]
    # There is a formula for this:
    # https://www.iap.uni - jena.de/iapmedia/de/Lecture/Advanced + Lens + Design1393542000/ALD13_Advanced + Lens + Design + 7 + _ + Aspheres + and +freeforms.pdf
    # but I obtained it numerically (in demo_conic_surface.py).
    kappa = 0.55555 # 0.55555
    f = roc*n2/(n2 - n1)
    surface = ZemaxConic(roc, 0.3, 1, kappa)
    element = SimpleElement(surface, UniformIsotropic(ri.FixedIndex(n2)), perfect_refractor)
    assembly = Assembly(surface, [element], UniformIsotropic(ri.FixedIndex(n1)))
    sphere_trace_kwargs = dict(epsilon=1e-9, t_max=1e9, max_steps=100)
    focus = (0, 0, f, 1)
    for x0 in x0s:
        ray = make_ray(x0, 0, -1, 0, 0, 1, 1, 0, 0, n1, 1, 0, 800e-9)
        segments = assembly.nonseq_trace(ray, sphere_trace_kwargs).flatten()
        assert len(segments) == 2
        assert v4.norm(segments[1].ray.line.pass_point(focus)[1].origin - focus) < 1e-6




