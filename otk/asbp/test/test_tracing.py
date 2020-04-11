# import numpy as np
# from pyfathom import cge
# import otk.rt.raytrace
# from otk import paraxial, asbp, ri, og
# from otk import rt as rt




# Keep for now
# def test_beam_tracing_graph(qtbot):
#     n = 1.5
#     lamb = 860e-9
#     w = 80e-3
#     f = 100e-3
#     waist = 20e-6
#     k = 2*np.pi/lamb
#     roc, center_thickness = paraxial.design_thick_spherical_transform_lens(n, w, f)
#     lens_surfaces = rt.make_spherical_lens_surfaces(roc, -roc, center_thickness, ri.FixedIndex(n))
#     z_fp = center_thickness/2 + w
#     surface1 = rt.Surface(rt.PlanarProfile(), matrix=rt.make_translation(0, 0, z_fp))
#
#     nodes = []
#
#     beam = asbp.Beam(asbp.PlaneProfile.make_gaussian(lamb, 1, waist, 160e-6, 64, z_waist=-z_fp))
#     nodes.append(lambda:beam)
#
#     for surface in lens_surfaces:
#         nodes.append(asbp.PropagationTo(surface))
#         nodes.append(asbp.Refraction(surface))
#
#     nodes.append(asbp.PropagationTo(surface1))
#
#     graph = cge.Graph()
#     graph.add_path(nodes)
#     graph.execute()
#     nodes[-1].incident_beam






