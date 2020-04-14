import numpy as np
from collections import defaultdict
from typing import Dict, Sequence, Iterable, Union
from PyQt5 import QtWidgets, QtCore
from .. import sdb
from ..sdb.glsl import gen_get_all_recursive
from .scalar import Assembly
from ..sdb.qt import SphereTraceRender, SphereTraceViewer
from .._utility import Delegate
from . import Element
from .. import v4
from ..qt import application
from ..sdb.qt import SphereTraceViewer

__all__ = ['view_assembly', 'application', 'view_elements']

def view_elements(elements: Sequence[Element], all_properties: Dict[sdb.Surface, Dict] = None,
    surface: sdb.Surface = None, default_edge_width: float = None, projection_type: str = 'orthographic', zhat: Sequence[float]=None):
    if all_properties is None:
        all_properties = {}
    all_properties = defaultdict(dict, all_properties)

    if surface is None:
        surface = sdb.UnionOp([e.surface for e in elements])

    epsilon = v4.norm(surface.get_aabb(np.eye(4)).size)*1e-3

    if default_edge_width is None:
        default_edge_width = epsilon*2

    for surface in surface.descendants():
        all_properties[surface].setdefault('edge_width', default_edge_width)

    if zhat is None:
        zhat = -v4.xhat

    sdb_glsl = gen_get_all_recursive(surface, all_properties)
    viewer = SphereTraceViewer(sdb_glsl)
    viewer.epsilon = epsilon
    size = viewer.size()
    aspect = size.height()/size.width()
    projection, eye_to_world = sdb.lookat_surface(surface, projection_type, zhat, aspect)
    viewer.projection = projection
    viewer.eye_to_world = eye_to_world
    viewer.show()
    return viewer

def view_assembly(a: Assembly, all_properties: Dict[sdb.Surface, Dict] = None, surface: sdb.Surface = None,
    default_edge_width: float = None, projection_type: str = 'orthographic', zhat: Sequence[float]=None):
    if surface is None:
        surface = a.surface
    return view_elements(a.elements.values(), all_properties, surface, default_edge_width, projection_type, zhat)



