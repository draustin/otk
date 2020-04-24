import itertools
from ctypes import c_void_p
import numpy as np
from typing import Sequence, Tuple, Iterable
from dataclasses import dataclass
from OpenGL import GL
from . import glsl, render

def link_program(vertex_source:str, fragment_source:str):
    vertex_shader = GL.glCreateShader(GL.GL_VERTEX_SHADER)
    GL.glShaderSource(vertex_shader, vertex_source)
    GL.glCompileShader(vertex_shader)
    #print(GL.glGetShaderInfoLog(vertex_shader))

    fragment_shader = GL.glCreateShader(GL.GL_FRAGMENT_SHADER)
    GL.glShaderSource(fragment_shader, fragment_source)
    GL.glCompileShader(fragment_shader)
    #print(GL.glGetShaderInfoLog(fragment_shader))
    #print(fragment_source)

    program = GL.glCreateProgram()
    GL.glAttachShader(program, vertex_shader)
    GL.glAttachShader(program, fragment_shader)
    GL.glLinkProgram(program)

    #print(GL.glGetProgramInfoLog(program))

    GL.glDetachShader(program, vertex_shader)
    GL.glDetachShader(program, fragment_shader)

    return program

class SphereTraceProgram:
    def __init__(self, program:int, position_buffer:int):
        self.program = program
        self.position_buffer = position_buffer

    def draw(self, eye_to_world: np.ndarray, eye_to_clip: np.ndarray, max_steps:float,
        epsilon:float, background_color:Sequence[float]):
        assert len(background_color) == 4

        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.position_buffer)
        loc = GL.glGetAttribLocation(self.program, "position")
        GL.glEnableVertexAttribArray(loc)
        # Last argument must be None. See
        # https://stackoverflow.com/questions/44781223/minimalist-pyopengl-example-in-pyqt5-application
        GL.glVertexAttribPointer(loc, 2, GL.GL_FLOAT, GL.GL_FALSE, 0, None)

        clip_to_world = np.linalg.inv(eye_to_clip).dot(eye_to_world)
        world_to_clip = np.linalg.inv(clip_to_world)

        GL.glUseProgram(self.program)
        viewport = GL.glGetFloatv(GL.GL_VIEWPORT)
        GL.glUniform4i(GL.glGetUniformLocation(self.program, "viewport"), *viewport)

        GL.glUniformMatrix4fv(GL.glGetUniformLocation(self.program, "clip_to_world"), 1, GL.GL_TRUE, clip_to_world)
        GL.glUniformMatrix4fv(GL.glGetUniformLocation(self.program, "world_to_clip"), 1, GL.GL_TRUE, world_to_clip)
        # Light comes from infinity behind the camera, meaning +z direction (camera looks along -z).
        GL.glUniform3f(GL.glGetUniformLocation(self.program, "light_direction"), eye_to_world[2, 0], eye_to_world[2, 1],
            eye_to_world[2, 2])
        GL.glUniform4f(GL.glGetUniformLocation(self.program, 'background_color'), *background_color)

        # https://stackoverflow.com/questions/33059185/qopenglwidgets-resizegl-is-not-the-place-to-call-glviewport
        GL.glUniform1i(GL.glGetUniformLocation(self.program, 'max_steps'), max_steps)
        GL.glUniform1f(GL.glGetUniformLocation(self.program, 'epsilon'), epsilon)
        GL.glDrawArrays(GL.GL_TRIANGLE_STRIP, 0, 4)

def setup_trace_position_buffer():
    position_array = np.asarray([(-1, -1), (-1, 1), (1, -1), (1, 1)], dtype=np.float32).ravel().view(np.ubyte)
    position_buffer = GL.glGenBuffers(1)
    GL.glBindBuffer(GL.GL_ARRAY_BUFFER, position_buffer)
    GL.glBufferData(GL.GL_ARRAY_BUFFER, position_array.nbytes, position_array, GL.GL_STATIC_DRAW)
    return position_buffer

def make_sphere_trace_program(sdb_glsl: str):
    fragment_source = glsl.sdf_glsl + sdb_glsl + glsl.trace_glsl
    program = link_program('#version 120\n\n' + glsl.trace_vertex_source, '#version 120\n\n' + fragment_source)
    position_buffer = setup_trace_position_buffer()
    return SphereTraceProgram(program, position_buffer)

class RayProgram:
    def __init__(self, program: int, max_num_points: int, point_buffer: int):
        self.program = program
        self.max_num_points = max_num_points
        self.point_buffer = point_buffer
        self.indices = []

    # TODO enable rays elements to be nx4 as well as nx3.
    def set_rays(self, rays: Sequence[np.ndarray], colors:Iterable=None):
        assert all(ray.ndim == 2 and ray.shape[1] == 3 for ray in rays)
        num_points = sum(ray.shape[0] for ray in rays)
        if num_points > self.max_num_points:
            raise ValueError(f'Number of points {num_points} is greater than max. number of points {self.max_num_points}.')
        if colors is None:
            colors = itertools.repeat((1, 0, 0))

        buffer_data = np.empty((num_points, 3), dtype=np.float32)
        first = 0
        indices = []
        for ray in rays:
            count = ray.shape[0]
            buffer_data[first:first+count, :] = ray
            indices.append((first, count))
            first += count

        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.point_buffer)
        GL.glBufferSubData(GL.GL_ARRAY_BUFFER, 0, buffer_data.nbytes, buffer_data.ravel().view(np.ubyte))

        self.indices = indices
        self.colors = colors

    def draw(self, world_to_clip:np.ndarray):
        GL.glUseProgram(self.program)

        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.point_buffer)
        loc = GL.glGetAttribLocation(self.program, "position")
        GL.glEnableVertexAttribArray(loc) # TODO which of these are needed each draw?
        GL.glVertexAttribPointer(loc, 3, GL.GL_FLOAT, GL.GL_FALSE, 0, None)

        GL.glUniformMatrix4fv(GL.glGetUniformLocation(self.program, "world_to_clip"), 1, GL.GL_TRUE, world_to_clip)
        color_loc = GL.glGetUniformLocation(self.program, 'color')
        for (first, count), color in zip(self.indices, self.colors):
            GL.glUniform3f(color_loc, color[0], color[1], color[2])
            GL.glDrawArrays(GL.GL_LINE_STRIP, first, count)


def make_ray_program(max_num_points:int):
    program = link_program('#version 120\n\n' + glsl.ray_vertex_source, '#version 120\n\n' + glsl.ray_fragment_source)

    point_buffer = GL.glGenBuffers(1)
    GL.glBindBuffer(GL.GL_ARRAY_BUFFER, point_buffer)
    GL.glBufferData(GL.GL_ARRAY_BUFFER, max_num_points*12, None, GL.GL_DYNAMIC_DRAW)

    return RayProgram(program, max_num_points, point_buffer)

@dataclass
class WireframeProgram:
    program: int
    vertex_buffer: int
    edge_buffer: int

    def __post_init__(self):
        self.num_models = 0

    def set_models(self, models: Sequence[render.WireframeModel]):
        self.models = models
        num_vertices = sum(s.num_vertices for s in models)
        num_edges = sum(s.num_edges for s in models)
        vertex_data = np.empty((num_vertices, 3), np.float32)
        edge_data = np.empty((num_edges, 2), np.uint32)
        first_vertex = 0
        first_edge = 0
        for model in models:
            last_vertex = first_vertex + model.num_vertices
            last_edge = first_edge + model.num_edges
            vertex_data[first_vertex:last_vertex, :] = model.vertices[:, :3]
            edge_data[first_edge:last_edge, :] = model.edges
            first_vertex = last_vertex
            first_edge = last_edge

        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.vertex_buffer)
        # TODO dynamic draw is correct flag?
        GL.glBufferData(GL.GL_ARRAY_BUFFER, vertex_data.nbytes, vertex_data.ravel().view(np.ubyte), GL.GL_DYNAMIC_DRAW)

        GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, self.edge_buffer)
        GL.glBufferData(GL.GL_ELEMENT_ARRAY_BUFFER, edge_data.nbytes, edge_data.ravel().view(np.ubyte), GL.GL_DYNAMIC_DRAW)

    def draw(self, world_to_clip: np.ndarray):
        GL.glUseProgram(self.program)

        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.vertex_buffer)
        position_loc = GL.glGetAttribLocation(self.program, "position")
        GL.glEnableVertexAttribArray(position_loc)

        GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, self.edge_buffer)

        color_loc = GL.glGetUniformLocation(self.program, 'color')

        GL.glUniformMatrix4fv(GL.glGetUniformLocation(self.program, "world_to_clip"), 1, GL.GL_TRUE, world_to_clip)

        first_vertex = 0
        first_edge = 0
        for model in self.models:
            #print(model, first_vertex, first_edge)
            GL.glUniform3f(color_loc, model.color[0], model.color[1], model.color[2])
            # Cast offsets to c_void_p - https://stackoverflow.com/questions/55479781/python-opengl-sending-data-to-shader
            GL.glVertexAttribPointer(position_loc, 3, GL.GL_FLOAT, GL.GL_FALSE, 0, c_void_p(first_vertex*12))
            GL.glDrawElements(GL.GL_LINES, model.num_edges*2, GL.GL_UNSIGNED_INT, c_void_p(first_edge* 8))
            first_vertex += model.num_vertices
            first_edge += model.num_edges

    vertex_source = """\
#version 120
uniform mat4 world_to_clip;

attribute vec3 position;
void main (void) {
    gl_Position = vec4(position, 1.0)*world_to_clip;
}\n\n"""

    fragment_source = """\
#version 120
uniform vec3 color;
void main() {
    gl_FragColor = vec4(color, 1);
}
"""
    @classmethod
    def make(cls):
        program = link_program(cls.vertex_source, cls.fragment_source)
        vertex_buffer = GL.glGenBuffers(1)
        edge_buffer = GL.glGenBuffers(1)

        return cls(program, vertex_buffer, edge_buffer)







