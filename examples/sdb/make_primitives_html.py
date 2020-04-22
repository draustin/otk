import numpy as np
from otk import sdb
from otk.sdb import demoscenes, webex
scene = demoscenes.make_primitives()
eye_to_world = sdb.lookat(scene.eye, scene.center)
projection = sdb.Perspective(np.pi/3, scene.z_near, scene.z_far)
#projection = sdb.Orthographic(scene.z_far*np.tan(np.pi/6), scene.z_far)
webex.gen_html('primitives.html', scene.sdb_glsl, eye_to_world, projection, 100, 1e-2, (1, 0, 0, 1))
