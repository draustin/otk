"""Sequential beam tracing."""
from dataclasses import dataclass
from typing import Tuple, Sequence, Iterable
from  .. import rt1
from .tracing import Beam

__all__ = ['trace_surfaces']

@dataclass
class BeamSegment:
    """
    Args:
        surfaces:
        beams: In global coordinates.
    """
    surfaces: Tuple[rt1.Surface, rt1.Surface]
    beams: Tuple[Beam, Beam]
    planarized_beam: Beam

    def __post_init__(self):
        assert all((beam is None) == (surface is None) for beam, surface in zip(self.surfaces, self.beams))

    def __str__(self):
        def get_name(surface):
            if surface is None:
                return 'None'
            else:
                return surface.name

        return f'({get_name(self.surfaces[0])} to {get_name(self.surfaces[1])}), {self.beam}'

    def connect(self, other):
        assert self.surfaces[1] is None
        assert other.surfaces[0] is None
        return BeamSegment((self.surfaces[0], other.surfaces[1]), (self.beams[0], other.beams[1]), other.planarized_beam)

    def __add__(self, other):
        return self.connect(other)

@dataclass
class BeamOnSurface:
    surface: rt1.Surface
    beams: Sequence[Tuple[str, Beam]]

def process_surface(beam, surface: rt1.Surface, mode_key: str, last_surface:rt1.Surface, last_beam:Beam) -> Tuple[BeamSegment, BeamOnSurface, Beam]:
    incident_beam, planarized_beam = beam.propagate_to_surface(surface)

    segment = BeamSegment((last_surface, surface), (last_beam, incident_beam), planarized_beam)

    beams = []
    beams.append(('incident', incident_beam.transform(surface.inverse_matrix)))

    if surface.boundary is not None:
        # TODO option to clip?
        beams.append(('boundaried', beams[-1][1].apply_boundary(surface.boundary)))

    if surface.mask is not None:
        beams.append(('masked', beams[-1][1].apply_boundary(surface.mask)))

    if surface.interface is not None:
        to_mode = beams[-1][1].make_interface_modes(surface.profile, surface.interface)
        beams.append((mode_key, to_mode(mode_key)))
    else:
        assert mode_key is None

    beam_on_surface = BeamOnSurface(surface, beams)

    beam = beams[-1][1].transform(surface.matrix)

    return segment, beam_on_surface, beam


def trace_surfaces(beam, surfaces: Iterable[rt1.Surface], mode_keys: Iterable[str]) -> Tuple[
    Sequence[BeamSegment], Sequence[BeamOnSurface]]:
    """Trace over a sequence of surfaces.

    TODO: generalize mode_keys to (fold:bool, mode_key:str) pairs to allow pure (planar, infinite) fold mirrors.

    Args:
        surfaces: The surfaces.
        mode_keys: The interface mode to use at each surface.

    Returns:
        The first segment is open ended at the start. The last may or may not be open ended, depending
            on if the final surface yields a beam.
    """
    last_surface = None
    last_beam = None  # beam existing last_surface
    segments = []
    beam_on_surfaces = []

    for surface, mode_key in zip(surfaces, mode_keys):
        segment, beam_on_surface, beam = process_surface(beam, surface, mode_key, last_surface, last_beam)

        segments.append(segment)
        beam_on_surfaces.append(beam_on_surface)

        last_beam = beam
        last_surface = surface

    segments.append(BeamSegment((last_surface, None), (last_beam, None), None))

    return segments, beam_on_surfaces