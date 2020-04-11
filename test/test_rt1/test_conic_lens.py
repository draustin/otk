import numpy as np
import otk.h4t
from otk import ri, rt1

def test_conic_surface():
    # Prove that with a conic surface we can focus parallel rays perfectly.
    n1 = 1
    n2 = 1.5
    f = 1
    roc = f*(n2 - n1)/n2
    x0s = [-0.2, -0.1, 0, 0.1, 0.2]
    # There is a formula for this:
    # https://www.iap.uni - jena.de/iapmedia/de/Lecture/Advanced + Lens + Design1393542000/ALD13_Advanced + Lens + Design + 7 + _ + Aspheres + and +freeforms.pdf
    # but I obtained it numerically (in demo_conic_surface.py).
    kappa = 0.55555
    f = roc*n2/(n2 - n1)
    interface = rt1.FresnelInterface(ri.FixedIndex(n1), ri.FixedIndex(n2))
    conic_surface = rt1.Surface(rt1.ConicProfile(roc, kappa), interface=interface)
    origin = rt1.stack_xyzw(x0s, 0, -1, 1)
    vector = rt1.stack_xyzw(0, 0, 1, 0)
    detector_surface = rt1.Surface(rt1.PlanarProfile(), matrix=otk.h4t.make_translation(0, 0, f))
    surfaces = conic_surface, detector_surface
    line = rt1.Line(origin, vector)
    pol = rt1.cross(line.vector, [0,1,0,0])
    ray = rt1.raytrace.Ray(line, pol, 1, 0, 860e-9, n1)
    segments = ray.trace_surfaces(surfaces, ['transmitted', 'incident'])[0]
    assert len(segments) == 3
    xy = segments[-1].ray.line.origin[..., :2]
    # The focusing is close (f number 2.5) but not quite perfect due to errors in the intersection solver.
    assert np.allclose(xy, 0, atol=2e-6)

