import numpy as np
from matplotlib import pyplot as plt
from otk.rt2 import *

surface = DifferenceOp(
    IntersectionOp((Sphere(3.0, (1.0, 0.0, 0.0)), Sphere(3.0, (-1.0,0.0, 0.0)))),
    InfiniteCylinder(0.5))

target = np.zeros((64, 64))

P = np.linalg.inv(lookat([0.5, 0.0, -5.0], [0.0, 0.0, 0.0])) @ orthographic(-4, 4, -4, 4, 1, 20)
invP = np.linalg.inv(P)

def shade(nx:float, ny:float):
    return shade_distance(nx, ny, invP, surface, 1e-3, 100)

raster(target, shade)

plt.imshow(target)
plt.show()
