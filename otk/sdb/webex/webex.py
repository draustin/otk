import os
import shutil
import itertools
from typing import Sequence, Iterable
from functools import singledispatch
import numpy as np
from textwrap import dedent
from .. import glsl, render

@singledispatch
def gen_js_expr(obj) -> str:
    raise NotImplementedError(obj)

@gen_js_expr.register
def _(obj: render.Orthographic):
    return f'new Orthographic({obj.half_width}, {obj.z_far})'

@gen_js_expr.register
def _(obj: render.Perspective):
    return f'new Perspective({obj.fov}, {obj.z_near}, {obj.z_far})'

def gen_js_mat4(m: np.ndarray) -> str:
    assert m.shape == (4, 4)
    return 'glMatrix.mat4.fromValues(' + ', '.join(str(e) for e in m.flatten(order='F')) + ')'

def gen_js_vec4(v: Sequence[float]) -> str:
    v = np.asarray(v, float)
    assert v.shape == (4,)
    return 'glMatrix.vec4.fromValues(' + ', '.join(str(e) for e in v) + ')'

def gen_js_vec3(v: Sequence[float]) -> str:
    v = np.asarray(v, float)
    assert v.shape == (3,)
    return 'glMatrix.vec3.fromValues(' + ', '.join(str(e) for e in v) + ')'

def gen_js_Float32Array(a: np.ndarray) -> str:
    assert a.ndim == 1
    return 'new Float32Array([' + ', '.join(str(e) for e in a) + '])'

def gen_js_ray(points: Sequence[Sequence[float]], color: Sequence[float]) -> str:
    points = np.asarray(points, float)
    assert points.ndim == 2
    assert points.shape[1] == 3
    color = np.asarray(color, float)
    assert color.shape == (3,)
    points_str = '[' + ', '.join(gen_js_vec3(point) for point in points) + ']'
    return f'{{points: {points_str}, color: {gen_js_Float32Array(color)}}}'

# TODO here and elsewhere - combine rays and colors.
def gen_html(filename: str, sdb_glsl: str, eye_to_world: Sequence[Sequence[float]], projection: render.Projection, max_steps: int,
    epsilon: float, background_color: Sequence[float], rays: Sequence[np.ndarray]=None, colors: Iterable=None):
    """Generate HTML & supporting files for displaying a scene with given signed distance bound GLSL.

    The GLSL containing getSDB0 and getColor0 is given by sdb_glsl.
    """
    eye_to_world = np.asarray(eye_to_world, float)
    assert eye_to_world.shape == (4, 4)
    background_color = np.asarray(background_color, float)
    assert background_color.shape == (4,)

    js_path = os.path.splitext(filename)[0] + '-js'
    os.makedirs(js_path, exist_ok=True)

    if rays is None:
        rays = []
    if colors is None:
        colors = itertools.repeat((1, 0, 0))

    ## TODO scene inside an object
    with open(os.path.join(js_path, 'scene.js'), 'wt') as f:
        f.write(f'// Generated scene data and shader codes for {filename}.\n')
        f.write(f'var eye_to_world = {gen_js_mat4(eye_to_world)};\n')
        f.write(f'var projection = {gen_js_expr(projection)};\n')
        f.write(f'var max_steps = {max_steps};\n')
        f.write(f'var epsilon = {epsilon};\n')
        f.write(f'var background_color = {gen_js_vec4(background_color)};\n')

        f.write('var trace_vertex_source = `\n')
        f.write(glsl.trace_vertex_source)
        f.write('`;\n')

        f.write('var sdf_glsl = `\n')
        f.write(glsl.sdf_glsl)
        f.write('`;\n')

        f.write('var trace_glsl = `\n')
        f.write(glsl.trace_glsl)
        f.write('`;\n')

        f.write('var sdb_glsl = `\n')
        f.write(sdb_glsl)
        f.write('`;\n')

        f.write('var ray_vertex_source = `\n')
        f.write(glsl.ray_vertex_source)
        f.write('`;\n')

        f.write('var ray_fragment_source = `\n')
        f.write(glsl.ray_fragment_source)
        f.write('`;\n')

        f.write('var rays = [' + ','.join(gen_js_ray(points, color) for points, color in zip(rays, colors)) + '];\n')


    shutil.copy(os.path.join(os.path.dirname(__file__), 'opengl.js'), js_path)
    shutil.copy(os.path.join(os.path.dirname(__file__), 'gl-matrix-min.js'), js_path)
    shutil.copy(os.path.join(os.path.dirname(__file__), 'gl-matrix-ext.js'), js_path)
    shutil.copy(os.path.join(os.path.dirname(__file__), 'main.js'), js_path)

    js_dir = os.path.basename(js_path)

    with open(filename, 'wt') as f:
        f.write(dedent(f"""\
    <!DOCTYPE html>

    <html>
    
    <head>
    <style>
      /* remove the border */
      body {{
        border: 0;
        margin: 0;
        background-color: white;
      }}
      /* make the canvas the size of the viewport */
      canvas {{
        width: 100vw;
        height: 100vh;
        display: block;
      }}
    </style>
    </head>

    <body>
    
    <script type="text/javascript" src="{js_dir}/gl-matrix-min.js"></script>
    <script type="text/javascript" src="{js_dir}/gl-matrix-ext.js"></script>
    <script type="text/javascript" src="{js_dir}/opengl.js"></script>
    <script type="text/javascript" src="{js_dir}/scene.js"></script>
    <script type="text/javascript" src="{js_dir}/main.js"></script>
    
    <canvas id="glscreen" ></canvas>
    
    </body>
    </html>\n\n"""))
