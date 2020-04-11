"""Hierarchical transformation matrices."""
from dataclasses import dataclass
from typing import Sequence

import otk.h4t

from . import rt1
import numpy as np

@dataclass
class Node:
    parent: 'Node'
    local_to_parent: np.ndarray = None
    label: str = None

    def __post_init__(self):
        self.children = []
        if self.parent is not None:
           self.parent.children.append(self)
        if self.local_to_parent is None:
            self.local_to_parent = np.eye(4)

    @property
    def local_to_global(self):
        if self.parent is None:
            m = np.eye(4)
        else:
            m = self.local_to_parent.dot(self.parent.local_to_global)
        return m

    def set_local_to_parent(self, m):
        self.local_to_parent = m

def make_singlet_nodes(thickness: float, parent_node: Node = None):
    # Make node for correct position of rear surface relative to front surface.
    ideal_rear_node = Node(parent_node, otk.h4t.make_translation(0, 0, thickness), 'rear surface')
    # Make child node for error on rear surface position - centration, thickness of the singlet, relative tilt.
    error_rear_node = Node(ideal_rear_node, label='error')

    return ideal_rear_node, error_rear_node


def make_singlet_sequence_nodes(spaces: Sequence[float], thicknesses: Sequence[float], parent_node: Node = None):
    """Returns ideal singlet nodes in case parent_node is not given."""
    assert len(spaces) + 1 == len(thicknesses)
    z = spaces[0]

    ideal_singlet_nodes = []
    for singlet_num, (space, thickness) in enumerate(zip(spaces, thicknesses)):
        # Make node for correct position of front surface.
        ideal_singlet_node = Node(parent_node, otk.h4t.make_translation(0, 0, z), f'singlet {singlet_num}')
        ideal_singlet_nodes.append(ideal_singlet_node)

        # Make child node for error on singlet position.
        singlet_node = Node(ideal_singlet_node, label='error')

        make_singlet_nodes(thickness, singlet_node)

    return ideal_singlet_nodes