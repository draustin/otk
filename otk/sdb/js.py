"""Display models in web pages using WebGL."""
import os
import shutil
from typing import Sequence
from functools import singledispatch
import numpy as np
from textwrap import dedent
from . import glsl, render

@singledispatch
def gen_js_expr(obj) -> str:
    raise NotImplementedError(obj)

@gen_js_expr.register
def _(obj: render.Orthographic):
    return f'new Orthographic({obj.half_width}, {obj.z_far})'

@gen_js_expr.register
def _(obj: render.Perspective):
    return f'new Perspective({obj.fov}, {obj.z_near}, {obj.z_far})'

@gen_js_expr.register
def _(obj: np.ndarray):
    if obj.shape == (4, 4):
        return 'glMatrix.mat4.fromValues(' + ', '.join(str(e) for e in obj.flatten(order='F')) + ')'
    elif obj.shape == (4,):
        return 'glMatrix.vec4.fromValues(' + ', '.join(str(e) for e in obj) + ')'
    else:
        raise NotImplementedError(obj)

def gen_html(sdb_glsl: str, eye_to_world: Sequence[Sequence[float]], projection: render.Projection, max_steps: int, epsilon: float, background_color: Sequence[float], filename: str):
    """Under construction.

    Next step is to adapt opengl.py to Javascript to enable camera movement.
    Would be nice for most of the Javascript to be in a separate file.
    """
    eye_to_world = np.asarray(eye_to_world, float)
    assert eye_to_world.shape == (4, 4)
    background_color = np.asarray(background_color, float)
    assert background_color.shape == (4,)

    js_path = os.path.splitext(filename)[0] + '-js'
    os.makedirs(js_path, exist_ok=True)
    with open(os.path.join(js_path, 'scene.js'), 'wt') as f:
        f.write(f'// Generated scene data and shader codes for {filename}.\n')
        f.write(f'var eye_to_world = {gen_js_expr(eye_to_world)};\n')
        f.write(f'var projection = {gen_js_expr(projection)};\n')
        f.write(f'var max_steps = {max_steps};\n')
        f.write(f'var epsilon = {epsilon};\n')
        f.write(f'var background_color = {gen_js_expr(background_color)};\n')
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

    shutil.copy(os.path.join(os.path.dirname(__file__), 'opengl.js'), js_path)
    shutil.copy(os.path.join(os.path.dirname(__file__), 'gl-matrix-min.js'), js_path)
    shutil.copy(os.path.join(os.path.dirname(__file__), 'gl-matrix-ext.js'), js_path)
    shutil.copy(os.path.join(os.path.dirname(__file__), 'main.js'), js_path)

    js_dir = os.path.basename(js_path)

    with open(filename, 'wt') as f:
        f.write(dedent(f"""\
    <!DOCTYPE html>

    <html>
    <body>
    
    <script type="text/javascript" src="{js_dir}/gl-matrix-min.js"></script>
    <script type="text/javascript" src="{js_dir}/gl-matrix-ext.js"></script>
    <script type="text/javascript" src="{js_dir}/opengl.js"></script>
    <script type="text/javascript" src="{js_dir}/scene.js"></script>
    <script type="text/javascript" src="{js_dir}/main.js"></script>
    
    <canvas id="glscreen"></canvas>
    
    </body>
    </html>\n\n"""))
