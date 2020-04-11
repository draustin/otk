import numpy as np
from collections import defaultdict
from typing import Dict, Sequence, Iterable
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
    surface: sdb.Surface = None, default_edge_width: float = None):
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

    sdb_glsl = gen_get_all_recursive(surface, all_properties)
    viewer = SphereTraceViewer(sdb_glsl)
    viewer.epsilon = epsilon
    viewer.show()
    return viewer

def view_assembly(a: Assembly, all_properties: Dict[sdb.Surface, Dict] = None):
    return view_elements(a.elements, all_properties, a.surface)



