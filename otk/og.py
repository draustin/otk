"""Tracing of rays and beams through optical graphs.

Immature module - I was never really happy with the structure. The basic idea was to generalize sequential ray tracing
to a graph allowing rays & beams to take different paths through a system.

# Development thoughts.

## Late 2018

Option 1.
Nodes are 'logical' regions of space. Edges are transmission or reflection at interfaces.

Option 2.
The computational graph is a generalization of the list of beams/segments returned by a sequential trace. Nodes are
operations on rays/PBGs/asbp beams. An operation takes input beams... hmm - this would work better if nodes are beams.

Option 3.
The calculation is represented by a computational graph. Nodes are beams (PBG or ASBP). Nodes may or not be computed
at any point. Edges are operations e.g. propagation
to a surface. At any given moment we have a set of (node, operation) active pairs.
An operation acts on a node. It creates more nodes and edges, and returns a set of (node, operation) active pairs.

Option 4.
The system is represented by a graph of transformers. Transformers accept a field as input and produce multiple new
fields, each with a key. The fields are sent to the transformers along the out edges. Can have more than one field
per key?

# 2019-04-22

Work in progress. Pre April 2019, the idea was for rays, bundles & beams to be subclasses of og.Field. og would
contain generic code for propagating fields through graphs of 'transformers' which operate on fields.

Upon reflection, don't like this. Imposing common base class for quite different notions. Strongly couples the development
of the optical graph logic and the underlying numeric stuff. Violates 'composition over inheritance'. Seems overkill
for simply ray tracing.

So decided to remove og.Field as common base class, and break og for now. Want the option of performing at nice simple
low-level interface.

"""
import copy
import numpy as np
from collections import deque, namedtuple, defaultdict
from typing import List, Tuple, Dict, Any, Set, Sequence
import networkx as nx
#from otk import pbg, asbp, rt

class Field:
    def __init__(self, lamb_mean: float, total_power: float):
        self.lamb_mean = lamb_mean
        self.total_power = total_power

    def propagate_to_surface(self, surface) -> Tuple['Field', Sequence, float]:
        raise NotImplementedError()

    def apply_interface(self, surface) -> Sequence:
        """

        Args:
            surface:

        Returns:
            Sequence (possibly empty) of (key, Field) pairs.
        """
        raise NotImplementedError()

    def apply_boundary(self, surface) -> 'Field':
        """

        Args:
            surface:

        Returns:
            New field with surface boundary applied, or self if no boundary, or None if field is blocked.
        """
        raise NotImplementedError()

    def apply_mask(self, surface) -> 'Field':
        """

        Args:
            surface:

        Returns:
            New field with mask applied, or self if no mask, or None if field is blocked.
        """
        raise NotImplementedError()

class Transformer:
    """Immutable object that transforms a field into (possibly multiple) new fields.

    The new fields are recorded in a graph. Each has a key, recorded as a property of the graph.

    Immutability is useful for lookup in dict/set/graph.
    """

    def apply(self, field) -> nx.DiGraph:
        """Transform field.

        Returns:
            Graph containing initial field, with 'key' property set as required.
        """
        raise NotImplementedError()

    def __hash__(self):
        return 0

    def __eq__(self, other):
        return type(self) is type(other)

ActivePair = namedtuple('ActivePair', ('field', 'transformer'))

class Tracer:
    def __init__(self, transformers: nx.DiGraph, fields: nx.DiGraph = None,
            active_pairs: Set[ActivePair] = None):
        assert all(isinstance(node, Transformer) for node in transformers)
        self.transformers = transformers
        if fields is None:
            fields = nx.DiGraph()
        assert all(isinstance(node, Field) for node in fields)
        self.fields = fields
        if active_pairs is None:
            active_pairs = set()
        active_pairs = {ActivePair(*p) for p in active_pairs}
        assert all(pair.field in fields for pair in active_pairs)
        self.active_pairs = active_pairs

    def transform_once(self):
        assert len(self.active_pairs) > 0
        transform(self.transformers, self.fields, self.active_pairs)

    def flush(self):
        while len(self.active_pairs) > 0:
            self.transform_once()

    def add_active_pair(self, field, transformer, key='initial'):
        self.fields.add_node(field, key=key)
        self.active_pairs.add(ActivePair(field, transformer))


def transform(transformer_graph: nx.DiGraph, field_graph: nx.DiGraph, actives: Set[Tuple[Field, Transformer]]) -> dict:
    # Pick next active pair.
    current_pair = None
    for pair in actives:
        if current_pair is None or pair.field.power > current_pair.field.power:
            current_pair = pair
    actives.remove(current_pair)
    current_field, current_transformer = current_pair
    assert current_field in field_graph

    new_field_graph = current_transformer.apply(current_field)
    # Sort fields by key.
    new_fields = defaultdict(list)
    for field in new_field_graph.nodes:
        key = new_field_graph.nodes[field]['key']
        new_fields[key].append(field)

    for node, data in new_field_graph.nodes(data=True):
        if node in field_graph.nodes:
            assert data['key'] == 'initial'
        else:
            field_graph.add_node(node, **data)

    field_graph.add_edges_from(new_field_graph.edges(data=True))

    for _, transformer, data in transformer_graph.out_edges(current_transformer, data=True):
        key = data['key']
        for field in new_fields[key]:
            actives.add((field, transformer))

class SurfaceTransformer(Transformer):
    """Transforms a Field at a Surface.

    Attributes:
        surface:
        operation: One of 'propagate', 'apply_boundary', 'mask' or 'deflect'.
    """
    def __init__(self, surface, operation: str):
        self.surface = surface
        self.operation = operation

    def __hash__(self):
        return hash((self.surface, self.operation))

    def __eq__(self, other):
        return type(self) is type(other) and self.surface is other.surface and self.operation == other.operation

    def apply(self, initial):
        graph = nx.DiGraph()
        graph.add_node(initial, key='initial')

        if self.operation == 'propagate':
            incident, intermediates, distance = initial.propagate_to_surface(self.surface)
            last_field = initial
            for key, field in intermediates:
                graph.add_node(field, key=key, surface=None)
                graph.add_edge(last_field, field, transformer=self)
                last_field = field
            graph.add_node(incident, key='incident', surface=self.surface)
            graph.add_edge(last_field, incident, transformer=self, distance=distance)
        elif self.operation == 'apply_boundary':
            after_boundary = initial.apply_boundary(self.surface)
            if after_boundary is initial:
                after_boundary = copy.copy(initial)
            graph.add_node(after_boundary, key='after_boundary', surface=self.surface)
            graph.add_edge(initial, after_boundary, transformer=self)
        elif self.operation == 'mask':
            masked = initial.apply_mask(self.surface)
            if masked is initial:
                masked = copy.copy(masked)
            graph.add_node(masked, key='masked', surface=self.surface)
            graph.add_edge(initial, masked, transformer=self, mask=self.surface.mask)
        elif self.operation == 'deflect':
            deflecteds = initial.apply_interface(self.surface)
            for key, field in deflecteds:
                graph.add_node(field, key=key, surface=self.surface)
                graph.add_edge(initial, field, transformer=self)

        return graph

    @classmethod
    def get_operation(cls, key):
        """Get operation string given beam key."""
        return {'incident':'propagate', 'after_boundary':'apply_boundary', 'masked':'mask'}.get(key, 'deflect')

    @classmethod
    def from_key(cls, surface, key):
        return cls(surface, cls.get_operation(key))

# class SurfaceTransformer(Transformer):
#     def __init__(self, surface):
#         self.surface = surface
#
#     def __hash__(self):
#         return hash(self.surface)
#
#     def __eq__(self, other):
#         return type(self) is type(other) and self.surface == other.surface
#
#     @classmethod
#     def from_key(cls, surface, key):
#         if key == 'incident':
#             node = PropagationToSurface(surface)
#         elif key == 'after_boundary':
#             node = SurfaceBoundary(surface)
#         elif key == 'masked':
#             node = SurfaceMask(surface)
#         else:
#             node = SurfaceDeflector(surface)
#         return node
#
# class PropagationToSurface(SurfaceTransformer):
#     def apply(self, initial:Field) -> nx.DiGraph:
#         """Returned graph contains field with key 'incident'."""
#         graph = nx.DiGraph()
#         graph.add_node(initial, key='initial')
#         incident, intermediates = initial.propagate_to_surface(self.surface)
#         last_field = initial
#         for key, field in intermediates:
#             graph.add_node(field, key=key)
#             graph.add_edge(last_field, field)
#             last_field = field
#         graph.add_node(incident, key='incident')
#         graph.add_edge(last_field, incident)
#         return graph
#
# class SurfaceBoundary(SurfaceTransformer):
#     def apply(self, initial:Field) -> nx.DiGraph:
#         graph = nx.DiGraph()
#         graph.add_node(initial, key='initial')
#
#         after_boundary = initial.apply_boundary(self.surface)
#         if after_boundary is not None:
#             if after_boundary is initial:
#                 after_boundary = copy.copy(initial)
#             graph.add_node(after_boundary, key='after_boundary')
#             graph.add_edge(initial, after_boundary)
#
#         return graph
#
# class SurfaceMask(SurfaceTransformer):
#     def apply(self, initial:Field) -> nx.DiGraph:
#         graph = nx.DiGraph()
#         graph.add_node(initial, key='initial')
#
#         masked = initial.apply_mask(self.surface)
#
#         if masked is not None:
#             if masked is initial:
#                 masked = copy.copy(masked)
#             graph.add_node(masked, key='masked', mask=self.surface.mask)
#             graph.add_edge(initial, masked)
#
#         return graph
#
# class SurfaceDeflector(SurfaceTransformer):
#
#     def apply(self, initial:Field) -> nx.DiGraph:
#         graph = nx.DiGraph()
#         graph.add_node(initial, key='initial')
#
#         deflecteds = initial.apply_interface(self.surface)
#         for key, field in deflecteds:
#             graph.add_node(field, key=key)
#             graph.add_edge(initial, field)
#
#         return graph

def make_surface_transformer_graph(surface):
    graph = nx.DiGraph()

    propagation = SurfaceTransformer(surface, 'propagate')
    graph.add_node(propagation)

    boundary = SurfaceTransformer(surface, 'apply_boundary')
    graph.add_node(boundary)
    graph.add_edge(propagation, boundary, key='incident')

    mask = SurfaceTransformer(surface, 'mask')
    graph.add_node(mask)
    graph.add_edge(boundary, mask, key='after_boundary')

    deflector = SurfaceTransformer(surface, 'deflect')
    graph.add_node(deflector)
    graph.add_edge(mask, deflector, key='masked')

    return graph

# class SurfaceTransformer(Transformer):
#     """Transformer """
#
#     def __init__(self, surface):
#         self.surface = surface
#
#     def apply(self, initial:Field):
#         graph = nx.DiGraph()
#         graph.add_node(self, key='initial')
#
#         if self.surface.is_fold():
#             raise NotImplementedError()
#         else:
#             incident, intermediates = initial.propagate_to_surface(self.surface)
#             last_field = initial
#             for key, field in intermediates:
#                 graph.add_node(field, key=key)
#                 graph.add_edge(last_field, field)
#                 last_field = field
#             graph.add_node(incident, key='incident')
#
#             after_boundary = last_field.apply_boundary(self.surface)
#             graph.add_node(after_boundary, key='after_boundary')
#             graph.add_edge(last_field, after_boundary)
#
#             deflecteds = after_boundary.apply_interface(self.surface)
#             for key, field in deflecteds:
#                 graph.add_node(field, key=key)
#                 graph.add_edge(after_boundary, field)
#
#         return graph


def make_surface_path(surfaces, keys):
    assert len(surfaces) == len(keys) + 1
    #transformers = [SurfaceTransformer(s) for s in surfaces]

    graph = nx.DiGraph()
    graph.update(make_surface_transformer_graph(surfaces[0]))

    for s0, s1, key in zip(surfaces[:-1], surfaces[1:], keys):
        graph.update(make_surface_transformer_graph(s1))
        n0 = SurfaceTransformer(s0, SurfaceTransformer.get_operation(key))
        n1 = SurfaceTransformer(s1, 'propagate')
        graph.add_edge(n0, n1, key=key)

    return graph

# def get_segments(fields:nx.DiGraph):
#     segments = []
#     for field in nx.topological_sort(fields):
#         for _, next_field, data in fields.out_edges(field, data=True):
#             if data['key'] == 'propagation':
#                 segments.append(Segment(field, data['distance']))
#     return segments

"""Contains a Field and its associated graph node data."""
FieldDataPair = namedtuple('FieldDataPair', ('field', 'data'))

def backtrace(graph:nx.DiGraph, field:Field) -> Tuple[List[FieldDataPair], List[dict]]:
    """

    Args:
        graph:
        field:
    """
    nodes = []
    edges = []
    while True:
        nodes.append(FieldDataPair(field, graph.nodes[field]))

        in_edges = graph.in_edges(field, data=True)
        if len(in_edges) == 0:
            break
        elif len(in_edges) == 1:
            field, _, data = list(in_edges)[0]
        else:
            raise ValueError('Fields (nodes) should only have one in edge.')
        edges.append(data)
    return nodes[::-1], edges[::-1]

def trace_surface_path(surfaces, keys, field)  -> Tuple[List[FieldDataPair], List[dict]]:
    """

    Args:
        surfaces:
        keys:
        field:
    """
    assert len(surfaces) == len(keys)
    graph = make_surface_path(surfaces, keys[:-1])
    tracer = Tracer(graph)
    tracer.add_active_pair(field, SurfaceTransformer(surfaces[0], 'propagate'))
    tracer.flush()
    # Find field with final key transformed by final surface.
    final_key = keys[-1]
    final_transformer = SurfaceTransformer.from_key(surfaces[-1], final_key)
    final_field = next(field for _, field, data in tracer.fields.edges(data=True) if tracer.fields.nodes[field]['key'] == final_key and data['transformer'] == final_transformer)
    nodes, edges = backtrace(tracer.fields, final_field)
    return nodes, edges

def connect_points(surfaces, keys, xyi_target, xyf_target, vi_start=None, max_error_distance=1e-9, max_num_iterations=100):
    """Find rays which connect two points through some surfaces.

    Args:
        surfaces (sequence of Surface objects): The initial point lies on the zeroth surface and the final point lies on
            the final surface.
        xyi_target: Local (x, y) coordinates of initial point on zeroth surface.
        xyf_target: Local (x, y) coordinates of final point on final surface.
        max_error_distance (scalar): The error tolerance.
        max_num_iterations (int): Maximum number of iterations allowed. For reasonable systems it seems to converge in
            only a handful of iterations.

    Returns:
        Tuple of (trace, error) tuples. The first is from initial to final and the second is the reverse. The trace
            is a sequence of intersections - returned by rt.trace. The error is the distance between the last intersection
            and the target.
        num_iterations (int): The number of iterations required for convergence.
    """
    assert len(surfaces) == len(keys) + 2
    ri_global = surfaces[0].to_global(rt.stack_xyzw(*xyi_target, 0, 1))
    rf_global = surfaces[-1].to_global(rt.stack_xyzw(*xyf_target, 0, 1))
    if vi_start is None:
        vi_start = (0, 0, 1, 0)
    vi_global = surfaces[0].to_global(vi_start)

    num_iterations = 0
    while num_iterations <= max_num_iterations:
        # Trace ray from xyi_target with direction vi_global.
        trace_if = trace_surface_path(surfaces[1:], keys, rt.Line(ri_global, vi_global))
        xf, yf, zf, _ = rt.to_xyzw(surfaces[-1].to_local(trace_if.intersections[-1].point))
        assert np.allclose(zf, 0)
        error_f = ((xf - xyf_target[0])**2 + (yf - xyf_target[1])**2)**0.5
        vf_global = -trace_if.intersections[-2].deflected_vector

        # Trace ray back xyf_target with direction vf_global.
        trace_fi = trace_surface_path(surfaces[-2::-1], keys[::-1], rt.Line(rf_global, vf_global))
        xi, yi, zi, _ = rt.to_xyzw(surfaces[0].to_local(trace_fi.intersections[-1].point))
        assert np.allclose(zi, 0)
        error_i = ((xi - xyi_target[0])**2 + (yi - xyi_target[1])**2)**0.5
        vi_global = -trace_fi.intersections[-2].deflected_vector

        error_distance = max(error_f.max(), error_i.max())
        if error_distance <= max_error_distance:
            break

        num_iterations += 1

    if num_iterations > max_num_iterations:
        raise ValueError('error_distance, %g, did not reach max_error_distance, %g, in %d iterations.'%(
        error_distance, max_error_distance, num_iterations))

    return ((trace_if, error_f), (trace_fi, error_i)), num_iterations