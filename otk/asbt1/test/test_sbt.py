import otk.h4t
import otk.rt1._lines
import otk.rt1._raytrace
import numpy as np
from otk import paraxial, asbp, ri, asbt1, v4hb
from otk import rt1


def test_beam_tracing(qtbot):
    n = 1.5
    lamb = 860e-9
    w = 80e-3
    f = 100e-3
    waist = 20e-6
    k = 2*np.pi/lamb
    roc, center_thickness = paraxial.design_thick_spherical_transform_lens(n, w, f)
    lens_surfaces = rt1.make_spherical_lens_surfaces(roc, -roc, center_thickness, ri.FixedIndex(n))
    z_fp = center_thickness/2 + w
    beam = asbt1.Beam(asbp.PlaneProfile.make_gaussian(lamb, 1, waist, 160e-6, 64, z_waist=-z_fp))
    surface1 = rt1.Surface(rt1.PlanarProfile(), otk.h4t.make_translation(0, 0, z_fp))
    surfaces = lens_surfaces + (surface1,)
    beam_segments = asbt1.trace_surfaces(beam, surfaces, ['transmitted', 'transmitted', None])

    origin = beam.to_global(v4hb.stack_xyzw([0, -waist, waist, 0, 0], [0, 0, 0, -waist, waist], -z_fp, 1))
    vector_local = v4hb.normalize(v4hb.stack_xyzw([0, 2/waist, -2/waist, 0, 0], [0, 0, 0, 2/waist, -2/waist], k, 0))
    vector = beam.to_global(vector_local)
    line = otk.rt1._lines.Line(origin, vector)
    polarization = v4hb.normalize(v4hb.cross(vector, [1,0,0,0]))
    ray_segments = otk.rt1._raytrace.Ray(line, polarization, 1, 0, lamb, 1).trace_surfaces(surfaces, ['transmitted', 'transmitted', 'incident'])

    # TODO resurrect
    # segments = [asbp.BeamRaySegment.combine(bs, rs) for bs, rs in zip(beam_segments, ray_segments)]
    #
    # widget = asbp.MultiProfileWidget.plot_segments(segments)
    # qtbot.addWidget(widget)
    # for n in range(len(widget.entries)):
    #     widget.set_index(n)