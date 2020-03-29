import otk.rt.lines
import otk.rt.raytrace
import numpy as np
from otk import paraxial, asbp, ri
from otk import rt as rt


def test_beam_tracing(qtbot):
    n = 1.5
    lamb = 860e-9
    w = 80e-3
    f = 100e-3
    waist = 20e-6
    k = 2*np.pi/lamb
    roc, center_thickness = paraxial.design_thick_spherical_transform_lens(n, w, f)
    lens_surfaces = rt.make_spherical_lens_surfaces(roc, -roc, center_thickness, ri.FixedIndex(n))
    z_fp = center_thickness/2 + w
    beam = asbp.Beam(asbp.PlaneProfile.make_gaussian(lamb, 1, waist, 160e-6, 64, z_waist=-z_fp))
    surface1 = rt.Surface(rt.PlanarProfile(), rt.make_translation(0, 0, z_fp))
    surfaces = lens_surfaces + (surface1,)
    beam_segments = asbp.trace_surfaces(beam, surfaces, ['transmitted', 'transmitted', None])

    origin = beam.to_global(rt.stack_xyzw([0, -waist, waist, 0, 0], [0, 0, 0, -waist, waist], -z_fp, 1))
    vector_local = rt.normalize(rt.stack_xyzw([0, 2/waist, -2/waist, 0, 0], [0, 0, 0, 2/waist, -2/waist], k, 0))
    vector = beam.to_global(vector_local)
    line = otk.rt.lines.Line(origin, vector)
    polarization = rt.normalize(rt.cross(vector, [1,0,0,0]))
    ray_segments = otk.rt.raytrace.Ray(line, polarization, 1, 0, lamb, 1).trace_surfaces(surfaces, ['transmitted', 'transmitted', 'incident'])

    # TODO resurrect
    # segments = [asbp.BeamRaySegment.combine(bs, rs) for bs, rs in zip(beam_segments, ray_segments)]
    #
    # widget = asbp.MultiProfileWidget.plot_segments(segments)
    # qtbot.addWidget(widget)
    # for n in range(len(widget.entries)):
    #     widget.set_index(n)