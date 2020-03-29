import itertools
import numpy as np
from typing import Sequence, Tuple, Iterable
from dataclasses import dataclass
from OpenGL import GL
from . import glsl

def link_program(vertex_source:str, fragment_source:str):
    vertex_shader = GL.glCreateShader(GL.GL_VERTEX_SHADER)
    GL.glShaderSource(vertex_shader, vertex_source)
    GL.glCompileShader(vertex_shader)
    print(GL.glGetShaderInfoLog(vertex_shader))

    fragment_shader = GL.glCreateShader(GL.GL_FRAGMENT_SHADER)
    GL.glShaderSource(fragment_shader, fragment_source)
    GL.glCompileShader(fragment_shader)
    print(GL.glGetShaderInfoLog(fragment_shader))
    print(fragment_source)

    program = GL.glCreateProgram()
    GL.glAttachShader(program, vertex_shader)
    GL.glAttachShader(program, fragment_shader)
    GL.glLinkProgram(program)

    print(GL.glGetProgramInfoLog(program))

    GL.glDetachShader(program, vertex_shader)
    GL.glDetachShader(program, fragment_shader)

    return program

class SphereTraceProgram:
    def __init__(self, program:int, position_buffer:int):
        self.program = program
        self.position_buffer = position_buffer

    def draw(self, eye_to_world: np.ndarray, eye_to_clip: np.ndarray, resolution:Tuple[int,int], max_steps:float,
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
        loc = GL.glGetUniformLocation(self.program, "iResolution")
        GL.glUniform2f(loc, *resolution)

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
    program = link_program(glsl.trace_vertex_source, fragment_source)
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
        GL.glEnableVertexAttribArray(loc)
        GL.glVertexAttribPointer(loc, 3, GL.GL_FLOAT, GL.GL_FALSE, 0, None)

        GL.glUniformMatrix4fv(GL.glGetUniformLocation(self.program, "world_to_clip"), 1, GL.GL_TRUE, world_to_clip)
        color_loc = GL.glGetUniformLocation(self.program, 'color')
        for (first, count), color in zip(self.indices, self.colors):
            GL.glUniform3f(color_loc, color[0], color[1], color[2])
            GL.glDrawArrays(GL.GL_LINE_STRIP, first, count)


def make_ray_program(max_num_points:int):
    program = link_program(glsl.ray_vertex_source, glsl.ray_fragment_source)

    point_buffer = GL.glGenBuffers(1)
    GL.glBindBuffer(GL.GL_ARRAY_BUFFER, point_buffer)
    GL.glBufferData(GL.GL_ARRAY_BUFFER, max_num_points*12, None, GL.GL_DYNAMIC_DRAW)

    return RayProgram(program, max_num_points, point_buffer)


