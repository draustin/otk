"""Discontinued - generate optics in POV ray tracer using Vapory library.

In mid-2019 I experimented with using POV-Ray to generate pretty renderings of optical setups. I started off using
https://github.com/Zulko/vapory, but realised it was too limited and switched to generating POV-Ray files directly using
the pov module.
"""
import numpy as np
import vapory as vp

from . import functions, trains

def make_circular_spherical_lens(roc1, roc2, thickness, radius, n, *args):
    sag1 = functions.calc_sphere_sag(roc1, radius)
    sag2 = functions.calc_sphere_sag(roc2, radius)

    sections = []
    surface1 = vp.Sphere([0, 0, roc1], abs(roc1))
    if roc1 > 0:
        sections.append(vp.Intersection(surface1, vp.Cylinder([0, 0, 0], [0, 0, sag1], radius)))
    elif roc1 < 0:
        sections.append(vp.Difference(vp.Cylinder([0, 0, sag1], [0, 0, 0], radius), surface1))

    sections.append(vp.Cylinder([0, 0, max(sag1, 0)], [0, 0, thickness + min(sag2, 0)], radius))

    surface2 = vp.Sphere([0, 0, thickness + roc2], abs(roc2))
    if roc2 < 0:
        sections.append(vp.Intersection(surface2, vp.Cylinder([0, 0, thickness + sag2], [0, 0, thickness], radius)))
    elif roc2 > 0:
        sections.append(vp.Difference(vp.Cylinder([0, 0, thickness], [0, 0, thickness+sag2], radius), surface2))

    lens = vp.Union(*sections, vp.Texture('T_Glass3'), vp.Interior('ior', n), *args)
    return lens

def make_square_spherical_lens(roc1: float, roc2: float, thickness: float, side_length: float, n: float, *args):
    radius = side_length/2**0.5
    sag1 = functions.calc_sphere_sag(roc1, radius)
    sag2 = functions.calc_sphere_sag(roc2, radius)
    hsl = side_length/2

    sections = []
    if np.isfinite(roc1):
        surface1 = vp.Sphere([0, 0, roc1], abs(roc1))
        if roc1 > 0:
            sections.append(vp.Intersection(surface1, vp.Box([-hsl, -hsl, 0], [hsl, hsl, sag1])))
        elif roc1 < 0:
            sections.append(vp.Difference(vp.Box([-hsl, -hsl, sag1], [hsl, hsl, 0]), surface1))

    sections.append(vp.Box([-hsl, -hsl, max(sag1, 0)], [hsl, hsl, thickness + min(sag2, 0)]))

    surface2 = vp.Sphere([0, 0, thickness + roc2], abs(roc2))
    if np.isfinite(roc2):
        if roc2 < 0:
            sections.append(vp.Intersection(surface2, vp.Box([-hsl, -hsl, thickness + sag2], [hsl, hsl, thickness])))
        elif roc2 > 0:
            sections.append(vp.Difference(vp.Box([-hsl, -hsl, thickness], [hsl, hsl, thickness+sag2]), surface2))

    lens = vp.Union(*sections, vp.Texture('T_Glass2'), vp.Interior('ior', n), *args)# , *args) vp.Texture( vp.Pigment( 'color', [1,0,1] ))
    return lens

def make_singlet(singlet: trains.Singlet, shape: str, lamb: float = None, *args):
    if shape == 'square':
        return make_square_spherical_lens(singlet.surfaces[0].roc, singlet.surfaces[1].roc, singlet.thickness, singlet.radius*2**0.5, singlet.n(lamb), *args)

def make_singlet_sequence(sequence: trains.SingletSequence, shape: str, lamb: float=None, *args):
    z = sequence.spaces[0]
    objects = []
    for singlet, space in zip(sequence.singlets, sequence.spaces[1:]):
        objects.append(make_singlet(singlet, shape, lamb, 'translate', [0, 0, z]))
        z += space
    return vp.Union(*objects, *args)

def make_train(train: trains.Train, shape: str, radii='equal', lamb: float=None, *args):
    objs = []
    z = train.spaces[0]
    for i1, i2, thickness in zip(train.interfaces[:-1], train.interfaces[1:], train.spaces[1:]):
        if radii == 'equal':
            radius = i1.radius
            assert i2.radius == radius
        elif radii == 'max':
            radius = max((i1.radius, i2.radius))
        else:
            radius = radii
        if shape == 'square':
            obj = make_square_spherical_lens(i1.roc, i2.roc, thickness, radius*2**0.5, i1.n2(lamb))
        objs.append(obj)
        z += thickness
    return vp.Union(*objs, *args)

