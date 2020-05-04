from collections import defaultdict
from typing import Dict, Sequence, Iterable

import numpy as np

from . import Element
from .. import sdb
from .. import v4h
from ..functions import norm
from ..sdb import webex
from ..sdb.glsl import gen_get_all_recursive


def gen_scene_html(filename: str, elements: Sequence[Element], all_properties: Dict[sdb.Surface, Dict] = None,
    surface: sdb.Surface = None, default_edge_width: float = None, projection_type: str = 'orthographic',
    zhat: Sequence[float]=None, rays: Sequence[np.ndarray] = None, colors: Iterable = None, epsilon: float=None,
    max_steps: int = 1000):
    if all_properties is None:
        all_properties = {}
    all_properties = defaultdict(dict, all_properties)

    if surface is None:
        surface = sdb.UnionOp([e.surface for e in elements])

    if epsilon is None:
        epsilon = norm(surface.get_aabb(np.eye(4)).size)*1e-3

    if default_edge_width is None:
        default_edge_width = epsilon*2

    for surface in surface.descendants():
        all_properties[surface].setdefault('edge_width', default_edge_width)

    if zhat is None:
        zhat = -v4h.xhat

    sdb_glsl = gen_get_all_recursive(surface, all_properties)

    # Don't know if aspect ratio of canvas is specified in advance, or whether it can be. Guess 1.0 for now.
    projection, eye_to_world = sdb.lookat_surface(surface, projection_type, zhat, 1.0)

    webex.gen_html(filename, sdb_glsl, eye_to_world, projection, max_steps, epsilon, (1, 1, 1, 1), rays, colors)

