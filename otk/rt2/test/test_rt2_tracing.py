import numpy as np
from otk import ri
from otk import functions
from otk import paraxial
from otk import sdb
from otk.sdb import npscalar
#from otk.sdb import *
#from otk.sdb.npscalar import *
from otk.functions import dot, normalize, norm
from otk.rt2.scalar import _get_transformed_element, _process_ray
from otk.rt2 import rt2_scalar as rt2
from otk.sdb import lens
from otk import v4h, functions

def test_tracing():
    normal = v4h.to_vector((0, 0, -1))
    constant = 0
    surface = sdb.Plane(normal, constant)
    deflector = rt2.make_fresnel_deflector()
    n0 = 1
    n0fun = ri.FixedIndex(n0)
    n1  = 1.5
    n1fun = ri.FixedIndex(n1)
    element = rt2.SimpleElement(surface, rt2.UniformIsotropic(n1fun), deflector)

    assert rt2.get_deflector(element, np.asarray((1, 2, 3, 1))) is deflector

    assembly = rt2.Assembly(element.surface, [element], rt2.UniformIsotropic(n0fun))
    assert _get_transformed_element(assembly, v4h.to_point((1, 2, -1))) is None
    assert _get_transformed_element(assembly, v4h.to_point((1, 2, 1))).element is element

    lamb = 800e-9
    incident_ray = rt2.make_ray(assembly, 1., 2, -1, 0, 1, 1, 1, 0, 0, lamb)
    epsilon = 1e-9
    length, deflected_rays = _process_ray(assembly, incident_ray, dict(epsilon=epsilon, t_max=1e9, max_steps=100))
    assert (2**0.5 - epsilon) <= length  <= (2**0.5 + epsilon)

    (rp, rs), (tp, ts) = functions.calc_fresnel_coefficients(n0, n1, abs(dot(incident_ray.line.vector, normal)))
    assert 0 <= npscalar.getsdb(surface, deflected_rays[0].line.origin) <= epsilon
    assert np.array_equal(deflected_rays[0].line.vector, normalize(v4h.to_vector((0, 1, -1, 0))))
    assert np.isclose(deflected_rays[0].flux, rs**2)

    vy = 2**-0.5*n0/n1
    assert 0 >= npscalar.getsdb(surface, deflected_rays[1].line.origin) >= -epsilon
    assert np.allclose(deflected_rays[1].line.vector, (0, vy, (1 - vy**2)**0.5, 0))
    assert np.isclose(deflected_rays[1].flux, ts**2*n1/n0)
    #assert np.array_equal(deflected_rays[0].vector, (0, 0, -1, 0))

def test_parabolic_mirror():
    roc = 0.1
    f = roc/2
    vertex = np.asarray((0., 0, 0))
    # Creater a parabolic mirror pointing along +z axis.
    surface = sdb.BoundedParaboloid(roc, 1.5*roc, True, vertex)
    deflector = rt2.make_constant_deflector(1, 1, 0, 0, True, False)
    n = ri.vacuum
    assembly = rt2.Assembly(surface, [rt2.SimpleElement(surface, rt2.UniformIsotropic(n), deflector)], rt2.UniformIsotropic(n))
    lamb = 800e-9
    epsilon = 1e-9
    focus = vertex + (0, 0, f)

    incident_ray = rt2.make_ray(assembly, *focus, 1, 1, 0, 1, -1, 0, lamb)
    length, deflected_rays = _process_ray(assembly, incident_ray, dict(epsilon=epsilon, t_max=1e9, max_steps=100))
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
    surface = lens.make_spherical_singlet(roc0, roc1, thickness, lens.make_circle(r), vertex0[:3])
    # Vertices are on surface.
    assert np.isclose(npscalar.getsdb(surface, vertex0), 0)
    assert np.isclose(npscalar.getsdb(surface, vertex1), 0)
    assert np.isclose(npscalar.getsdb(surface, vertex0 + (0, 0, -thickness, 0)), thickness)


    element = rt2.SimpleElement(surface, rt2.UniformIsotropic(n), rt2.perfect_refractor)
    assembly = rt2.Assembly(surface, [element], rt2.UniformIsotropic(ne))
    sphere_trace_kwargs = dict(epsilon=1e-9, t_max=1e9, max_steps=100)

    # Compute object and image points for 2f-2f imaging. zp is relative to vertex.
    zp_object = -(2*f - h0)
    zp_image = 2*f + h1 + thickness
    object = vertex0 + (0, 0, zp_object, 0)
    image = vertex0 + (0, 0, zp_image, 0)

    # Trace on-axis ray from object point.
    incident_ray0 = rt2.make_ray(assembly, *object[:3], 0, 0, 1, 0, 1, 0, lamb)
    branch = rt2.nonseq_trace(assembly, incident_ray0, sphere_trace_kwargs)
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
    incident_ray1 = rt2.make_ray(assembly, *object[:3], 0.006, 0, 1, 0, 1, 0, lamb)
    segments = rt2.nonseq_trace(assembly, incident_ray1, sphere_trace_kwargs).flatten()
    assert len(segments) == 3
    # Check ray passes close to image.
    assert norm(segments[2].ray.line.pass_point(image)[1].origin - image) < 1e-6

    # TODO check phase

def test_hyperbolic_lens():
    # Prove that with a conic surface we can focus parallel rays perfectly.
    n1 = 1
    n2 = 1.5
    f = 1
    roc = f*(n2 - n1)/n2
    x0s = [-0.2, -0.1, 0., 0.1, 0.2]
    # There is a formula for this:
    # https://www.iap.uni - jena.de/iapmedia/de/Lecture/Advanced + Lens + Design1393542000/ALD13_Advanced + Lens + Design + 7 + _ + Aspheres + and +freeforms.pdf
    # but I obtained it numerically (in demo_conic_surface.py).
    kappa = 0.55555 # 0.55555
    f = roc*n2/(n2 - n1)
    surface = sdb.ZemaxConic(roc, 0.3, 1, kappa)
    element = rt2.SimpleElement(surface, rt2.UniformIsotropic(ri.FixedIndex(n2)), rt2.perfect_refractor)
    assembly = rt2.Assembly(surface, [element], rt2.UniformIsotropic(ri.FixedIndex(n1)))
    sphere_trace_kwargs = dict(epsilon=1e-9, t_max=1e9, max_steps=100)
    focus = (0, 0, f, 1)
    for x0 in x0s:
        ray = rt2.make_ray(assembly, x0, 0, -1, 0, 0, 1, 1, 0, 0, 800e-9)
        segments = rt2.nonseq_trace(assembly, ray, sphere_trace_kwargs).flatten()
        assert len(segments) == 2
        assert norm(segments[1].ray.line.pass_point(focus)[1].origin - focus) < 1e-6




