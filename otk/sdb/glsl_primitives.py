"""Shader code generation for primitives."""
import numpy as np
from textwrap import dedent
from typing import Mapping
from . import *
from .glsl import get_property

__all__ = []

@gen_getSDB.register
def _(s:Plane, ids:Mapping) -> str:
    id = ids[s]
    return wrap_sdb_expr(f'dot(x.xyz, {gen_vec3(s.n)}) + {s.c}', id)

@gen_getSDB.register
def _(s:Sphere, ids:Mapping) -> str:
    id = ids[s]
    return wrap_sdb_expr(f'sphereSDF({gen_vec3(s.o)}, {s.r}, x.xyz)', id)

@gen_getSDB.register
def _(s:Box, ids:Mapping) -> str:
    id = ids[s]
    # https://www.iquilezles.org/www/articles/distfunctions/distfunctions.htm
    return dedent(f"""\
        float getSDB{id:d}(in vec4 x) {{
            vec3 q = abs(x.xyz - {gen_vec3(s.center)}) - {gen_vec3(s.half_size - s.radius)};
            return length(max(q,0.0)) + min(max(q.x,max(q.y,q.z)),0.0) - {s.radius};
        }}\n\n""")

@gen_getSDB.register
def _(s:Torus, ids:Mapping) -> str:
    id = ids[s]
    # https://www.iquilezles.org/www/articles/distfunctions/distfunctions.htm
    return dedent(f"""\
        float getSDB{id:d}(in vec4 x) {{
            vec3 xp = x.xyz - {gen_vec3(s.center)};
            vec2 q = vec2(length(xp.xy) - {s.major}, xp.z);
            return length(q) - {s.minor};
        }}\n\n""")

@gen_getSDB.register
def _(s:Ellipsoid, ids:Mapping) -> str:
    id = ids[s]
    # https://www.iquilezles.org/www/articles/distfunctions/distfunctions.htm
    # Numerical innacuracy at large distances???
    return dedent(f"""\
        float getSDB{id:d}(in vec4 x) {{
            vec3 xp = x.xyz - {gen_vec3(s.center)};
            float k0 = length(xp / {gen_vec3(s.radii)});
            float k1 = length(xp / {gen_vec3(s.radii**2)});
            return k0*(k0 - 1.)/k1;
        }}\n\n""")

@gen_getSDB.register
def _(s:InfiniteCylinder, ids:Mapping) -> str:
    id = ids[s]
    return wrap_sdb_expr(f'circleSDF({gen_vec2(s.o)}, {s.r}, x.xy)', id)

@gen_getSDB.register
def _(s:SphericalSag, ids:Mapping) -> str:
    id = ids[s]
    if np.isfinite(s.roc):
        inside = s.side*np.sign(s.roc)
        op = 'min' if inside > 0 else 'max'
        return wrap_sdb_expr(f'{op}({inside}*(length(x.xyz - {gen_vec3(s.center)}) - {abs(s.roc)}), {-s.side}*(x.z - {s.center[2]}))', id)
    else:
        return wrap_sdb_expr(f'{-s.side}*(x.z - {s.vertex[2]})', id)

@gen_getSDB.register
def _(s:ToroidalSag, ids:Mapping) -> str:
    id = ids[s]
    insides = s.side*np.sign(s.rocs)
    ops = ['min' if inside > 0 else 'max' for inside in insides]
    #op = 'min' if insides[0] > 0 else 'max'
    return dedent(f"""\
        float getSDB{id:d}(in vec4 x) {{
            // xp is relative revolution axis
            vec3 xp = x.xyz - {gen_vec3(s.center)};
            vec2 q = vec2(xp.x, length(xp.yz) - {s.ror});
            float d_torus = {insides[0]}*(length(q) - {abs(s.rocs[0])});
            return {ops[1]}({ops[0]}(d_torus, {insides[1]}*q.y), {-s.side}*xp.z);
        }}\n\n""")

@gen_getSDB.register
def _(s:ZemaxConic, ids:Mapping) -> str:
    id = ids[s]
    string = dedent(f"""\
        float getSDB{id:d}(in vec4 x) {{
            float rho = min(length(x.xy - {gen_vec2(s.vertex[:2])}), {s.radius});
            float z = {s.vertex[2]};\n""")
    if np.isfinite(s.roc):
        string += f'    z += rho*rho/({s.roc}*(1. + sqrt(1. - {s.kappa}*rho*rho/{s.roc**2})));\n'
    if len(s.alphas) > 0:
        string += f'    float h = {s.alphas[-1]};\n'
        string += ''.join(f'    h = h*rho + {alpha};\n' for alpha in s.alphas[-2::-1])
        string += '    z += h*rho*rho;\n'
    string += dedent(f"""\
        return {s.side}*(z - x.z)/{s.lipschitz};
    }}\n\n""")
    return string

@gen_getSDB.register
def _(s:Sag, ids:Mapping) -> str:
    id = ids[s]
    sag_glsl = gen_getSag(s.sagfun, str(id))

    return sag_glsl + dedent(f"""\
        float getSDB{id}(in vec4 x) {{
            float sag = getSag{id}(x.xy - {gen_vec2(s.origin[:2])});
            return {s.side}*(sag + {s.origin[2]} - x.z)/{s.lipschitz};
        }}\n\n""")

@gen_getColor.register(Plane)
@gen_getColor.register(Sphere)
@gen_getColor.register(Box)
@gen_getColor.register(Torus)
@gen_getColor.register(Ellipsoid)
@gen_getColor.register(InfiniteCylinder)
@gen_getColor.register(SphericalSag)
@gen_getColor.register(ZemaxConic)
@gen_getColor.register(Sag)
def _(surface, ids:Mapping[Surface, int], properties:Mapping) -> str:
    color = get_property(properties, 'surface_color')
    id = ids[surface]
    return dedent(f"""\
        vec3 getColor{id}(in vec4 x) {{
            return {gen_vec3(color)};
        }}\n\n""")