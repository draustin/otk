from collections import defaultdict
import os
from textwrap import dedent
from typing import Mapping, Set, Dict
from functools import singledispatch
from . import *

__all__ = ['gen_getSDB', 'gen_getSDB_recursive', 'gen_getColor_recursive', 'add_ids', 'gen_get_all_recursive', 'wrap_sdb_expr',
    'gen_vec2', 'gen_vec3', 'gen_vec4', 'gen_getColor']

with open(os.path.join(os.path.dirname(__file__), 'sdf.glsl'), 'rt') as f:
    sdf_glsl = f.read()

with open(os.path.join(os.path.dirname(__file__), 'trace.glsl'), 'rt') as f:
    trace_glsl = f.read()

trace_vertex_source = """\
#if __VERSION__ == 300
    in vec2 position;
#else
    attribute vec2 position;
#endif

void main (void)
{
    gl_Position = vec4(position, 0.0, 1.0);
}
"""

# TODO putting combining these into RayProgram.
ray_vertex_source = """
uniform mat4 world_to_clip;

#if __VERSION__ == 300
    in vec3 position;
#else
    attribute vec3 position;
#endif

void main (void) {
    gl_Position = vec4(position, 1.0)*world_to_clip;
}
"""

ray_fragment_source = """
#if __VERSION__ == 300
    out vec4 FragColor;
#endif
uniform vec3 color;
void main() {
    #if __VERSION__ == 300
        FragColor
    #else
        gl_FragColor
    #endif
    = vec4(color, 1);
}
"""

# Edge detection is iffy at present. Set default to off.
default_properties = dict(edge_width=0, edge_color=(0, 0, 0), surface_color=(0, 0, 1))

def get_property(properties, name):
    return properties.get(name, default_properties[name])

def gen_vec2(x):
    return f'vec2({float(x[0])}, {float(x[1])})'

def gen_vec3(x):
    return f'vec3({float(x[0])}, {float(x[1])}, {float(x[2])})'

def gen_vec4(x):
    return f'vec4({float(x[0])}, {float(x[1])}, {float(x[2])}, {float(x[3])})'

def gen_mat4(x):
    return 'mat4(' + ', '.join(gen_vec4(x[:, c]) for c in range(4)) + ')'

def gen_bool(x):
    return 'true' if x else 'false'

def wrap_sdb_expr(expr:str, id:int):
    return dedent(f"""\
        float getSDB{id:d}(in vec4 x) {{
            return {expr:s};
        }}\n\n""")

@singledispatch
def gen_getSDB(s:Surface, ids:Mapping) -> str:
    raise  NotImplementedError()

@singledispatch
def gen_getColor(surface:Surface, ids:Mapping, properties:Mapping) -> str:
    raise NotImplementedError()

def gen_getNormal(id) -> str:
    return f"""\
vec4 getNormal{id}(in vec4 x) {{
    const float h = 0.00001; // TODO better determination of this
    const vec3 k = vec3(1,-1,0.);
    return normalize(k.xyyz*getSDB{id}( x + k.xyyz*h ) +
                     k.yyxz*getSDB{id}( x + k.yyxz*h ) +
                     k.yxyz*getSDB{id}( x + k.yxyz*h ) +
                     k.xxxz*getSDB{id}( x + k.xxxz*h ) );
}}\n\n"""

@gen_getSDB.register
def _(s:UnionOp, ids:Mapping) -> str:
    id = ids[s]
    return (dedent(f"""\
        float getSDB{id:d}(in vec4 x) {{
            float dp;
            float d = getSDB{ids[s.surfaces[0]]}(x);\n""") +
        ''.join(f'    dp = getSDB{ids[sp]:d}(x);\n    if (dp < d) d = dp;\n' for sp in s.surfaces[1:]) +
        dedent(f"""\
            return d;
        }}\n\n"""))

@gen_getColor.register
def _(s:UnionOp, ids:Mapping, properties:Mapping) -> str:
    id = ids[s]
    edge_width = get_property(properties, 'edge_width')
    edge_color = get_property(properties, 'edge_color')
    return (f"""\
vec3 getColor{id}(in vec4 x) {{
    float dp;
    vec4 normalp;
    float costheta;
    vec3 color = getColor{ids[s.surfaces[0]]}(x);
    float d = getSDB{ids[s.surfaces[0]]}(x);
    vec4 normal = getNormal{ids[s.surfaces[0]]}(x);\n""" +
        ''.join(f"""\
    dp = getSDB{ids[sp]:d}(x);
    normalp = getNormal{ids[sp]}(x);
    costheta = dot(normal, normalp);
    if (abs(dp) + abs(d) < {edge_width}*sqrt(1. - costheta*costheta))
        color = {gen_vec3(edge_color)};
    else if (dp < d)
        color = getColor{ids[sp]}(x);
    if (dp < d) {{
        d = dp;
        normal = normalp;
    }}\n""" for sp in s.surfaces[1:]) +
    """\
    return color;
}\n\n""")

@gen_getSDB.register
def _(s:IntersectionOp, ids:Mapping) -> str:
    id = ids[s]
    return (dedent(f"""\
        float getSDB{id:d}(in vec4 x) {{
            float dp;
            float d = getSDB{ids[s.surfaces[0]]}(x);\n""") +
        ''.join(f'    dp = getSDB{ids[sp]:d}(x);\n    if (dp > d) d = dp;\n' for sp in s.surfaces[1:]) +
        dedent(f"""\
            return d;
        }}\n\n"""))

@gen_getColor.register
def _(s:IntersectionOp, ids:Mapping, properties:Mapping) -> str:
    id = ids[s]
    edge_width = get_property(properties, 'edge_width')
    edge_color = get_property(properties, 'edge_color')

    return (f"""\
vec3 getColor{id}(in vec4 x) {{
    float dp;
    vec4 normalp;
    float costheta;
    vec3 color = getColor{ids[s.surfaces[0]]}(x);
    float d = getSDB{ids[s.surfaces[0]]}(x);
    vec4 normal = getNormal{ids[s.surfaces[0]]}(x);\n""" +
        ''.join(f"""\
    dp = getSDB{ids[sp]:d}(x);
    normalp = getNormal{ids[sp]}(x);
    costheta = dot(normal, normalp);
    if (abs(dp) + abs(d) < {edge_width}*sqrt(1. - costheta*costheta))
        color = {gen_vec3(edge_color)};
    else if (dp > d)
        color = getColor{ids[sp]}(x);
    if (dp > d) {{
        d = dp;
        normal = normalp;
    }}\n""" for sp in s.surfaces[1:]) +
    """\
    return color;
}\n\n""")

@gen_getSDB.register
def _(s:DifferenceOp, ids:Mapping) -> str:
    id = ids[s]
    return dedent(f"""\
        float getSDB{id:d}(in vec4 x) {{
            return max(getSDB{ids[s.surfaces[0]]}(x), -getSDB{ids[s.surfaces[1]]}(x));
        }}\n\n""")

@gen_getColor.register
def _(s:DifferenceOp, ids:Mapping, properties:Mapping) -> str:
    id = ids[s]
    id0 = ids[s.surfaces[0]]
    id1 = ids[s.surfaces[1]]
    edge_width = get_property(properties, 'edge_width')
    edge_color = get_property(properties, 'edge_color')
    return dedent(f"""\
        vec3 getColor{id:d}(in vec4 x) {{
            float d0 = getSDB{id0}(x);
            float d1 = getSDB{id1}(x);
            float dd = d0 + d1;
            float costheta = dot(getNormal{id0}(x), getNormal{id1}(x));
            if (dd < {-edge_width})
                return getColor{id1}(x);
            else if (abs(d0) + abs(d1) <= {edge_width}*sqrt(1. - costheta*costheta))
                return {gen_vec3(edge_color)};
            else
                return getColor{id0}(x);
        }}\n\n""")

@gen_getSDB.register
def _(s:Hemisphere, ids:Mapping) -> str:
    id = ids[s]
    return dedent(f"""\
        float getSDB{id:d}(in vec4 x) {{
            return min(sphereSDF({gen_vec3(s.o)}, {s.r}, x.xyz), {s.sign}*(x.z - {s.o[2]}));
        }}\n\n""")

@gen_getSDB.register
def _(s:InfiniteRectangularPrism, ids:Mapping) -> str:
    id = ids[s]
    return wrap_sdb_expr(f'rectangleSDF({gen_vec2(s.center)}, {gen_vec2((s.width/2, s.height/2))}, x.xy)', id)

@gen_getColor.register
def _(s:InfiniteRectangularPrism, ids:Dict[Surface, int], properties:Mapping) -> str:
    id = ids[s]
    edge_width = get_property(properties, 'edge_width')
    edge_color = get_property(properties, 'edge_color')
    surface_color = get_property(properties, 'surface_color')
    return dedent(f"""\
        vec3 getColor{id}(in vec4 x) {{
            vec2 xp = x.xy - {gen_vec2(s.center)};
            vec2 q = abs(xp) - {gen_vec2((s.width/2, s.height/2))};
            if (length(q) < {edge_width})
                return {gen_vec3(edge_color)};
            else
                return {gen_vec3(surface_color)};
        }}\n\n""")

@gen_getSDB.register
def _(surface:AffineOp, ids:Mapping) -> str:
    id = ids[surface]
    child_id = ids[surface.surfaces[0]]
    return dedent(f"""\
        float getSDB{id:d}(in vec4 x) {{
            vec4 xp = x*{gen_mat4(surface.invm)};
            return getSDB{child_id:d}(xp)*{surface.scale};
        }}\n\n""")

@gen_getColor.register
def _(surface:AffineOp, ids:Dict[Surface, int], properties:Mapping) -> str:
    id = ids[surface]
    id_child = ids[surface.surfaces[0]]
    return dedent(f"""\
        vec3 getColor{id}(in vec4 x) {{
            return getColor{id_child}(x*{gen_mat4(surface.invm)});
        }}\n\n""")

@gen_getSDB.register
def _(surface:FiniteRectangularArray, ids:Mapping) -> str:
    id = ids[surface]
    child_id = ids[surface.surfaces[0]]
    return dedent(f"""\
        float getSDB{id:d}(in vec4 x) {{
            vec2 indices = clamp(floor((x.xy - {gen_vec2(surface.corner)})/{gen_vec2(surface.pitch)}), vec2(0., 0.), {gen_vec2(surface.size - 1)});
            vec2 center = (indices + 0.5)*{gen_vec2(surface.pitch)} + {gen_vec2(surface.corner)};
            return getSDB{child_id:d}(vec4(x.xy - center, x.zw));
        }}\n\n""")

@gen_getColor.register(FiniteRectangularArray)
def _(surface, ids:Dict[Surface, int], properties:Mapping) -> str:
    id = ids[surface]
    child_id = ids[surface.surfaces[0]]
    return dedent(f"""\
        vec3 getColor{id}(in vec4 x) {{
            vec2 indices = clamp(floor((x.xy - {gen_vec2(surface.corner)})/{gen_vec2(surface.pitch)}), vec2(0., 0.), {gen_vec2(surface.size - 1)});
            vec2 center = (indices + 0.5)*{gen_vec2(surface.pitch)} + {gen_vec2(surface.corner)};
            return getColor{child_id:d}(vec4(x.xy - center, x.zw));
        }}\n\n""")


@gen_getSDB.register
def _(s:SegmentedRadial, ids:Mapping) -> str:
    id = ids[s]
    return (dedent(f"""\
        float getSDB{id:d}(in vec4 x) {{
            float rho = length(x.xy - {gen_vec2(s.vertex)});
            if (rho <= {s.radii[0]}) return getSDB{ids[s.surfaces[0]]}(x);\n""") +
        '\n'.join(f'    else if (rho <= {r}) return getSDB{ids[s]}(x);' for r, s in zip(s.radii[1:], s.surfaces[1:-1])) +
        f'    else return getSDB{ids[s.surfaces[-1]]}(x);' +
        '}\n\n')

@gen_getColor.register
def _(s:SegmentedRadial, ids:Mapping, properties:Mapping) -> str:
    id = ids[s]
    return (dedent(f"""\
        vec3 getColor{id}(in vec4 x) {{
            float rho = length(x.xy - {gen_vec2(s.vertex)});
            if (rho <= {s.radii[0]}) return getColor{ids[s.surfaces[0]]}(x);\n""") +
        '\n'.join(f'    else if (rho <= {r}) return getColor{ids[s]}(x);' for r, s in zip(s.radii[1:], s.surfaces[1:-1])) +
        f'    else return getColor{ids[s.surfaces[-1]]}(x);' +
        '}')

@singledispatch
def gen_getSDB_recursive(surface:Surface, ids:Mapping[Surface,int], done:Set[int]) -> str:
    # Postcondition: ID of surface and all its children are in done.
    raise NotImplementedError()

@gen_getSDB_recursive.register
def _(surface:Compound, ids:Mapping[Surface,int], done:Set[int]=None) -> str:
    if done is None:
        done = set()
    id = ids[surface]
    assert id not in done
    code = ''
    for child_surface in surface.surfaces:
        child_id = ids[child_surface]
        if child_id not in done:
            code += gen_getSDB_recursive(child_surface, ids, done)
    code += gen_getSDB(surface, ids)
    done.add(id)
    return code

@gen_getSDB_recursive.register
def _(surface:Primitive, ids:Mapping[Surface,int], done:Set[int]) -> str:
    if done is None:
        done = set()
    code = gen_getSDB(surface, ids)
    done.add(ids[surface])
    return code

@singledispatch
def gen_getColor_recursive(surface:Surface, ids:Mapping[Surface,int], all_properties:Dict[Surface, Dict], done:Set[int]=None) -> str:
    raise NotImplementedError()

@gen_getColor_recursive.register
def _(surface:Compound, ids, all_properties, done=None) -> str:
    if done is None:
        done = set()
    id = ids[surface]
    assert id not in done
    code = ''
    for child_surface in surface.surfaces:
        child_id = ids[child_surface]
        if child_id not in done:
            code += gen_getNormal(ids[child_surface])
            code += gen_getColor_recursive(child_surface, ids, all_properties, done)
    code += gen_getColor(surface, ids, all_properties.get(surface, {}))
    done.add(id)
    return code

@gen_getColor_recursive.register
def _(surface:Primitive, ids, all_properties, done=None) -> str:
    if done is None:
        done = set()
    code = gen_getColor(surface, ids, all_properties.get(surface, {}))
    done.add(ids[surface])
    return code

@singledispatch
def add_ids(surface:Surface, ids:Dict[Surface, int]):
    raise NotImplementedError()

@add_ids.register
def _(surface:Primitive, ids:Dict[Surface, int]=None):
    if ids is None:
        ids = {}
    if surface not in ids:
        ids[surface] = max(ids.values(), default=-1) + 1
    return ids

@add_ids.register
def _(surface:Compound, ids:Dict[Surface, int]=None):
    if ids is None:
        ids = {}
    if surface not in ids:
        ids[surface] = max(ids.values(), default=-1) + 1
    for child_surface in surface.surfaces:
        add_ids(child_surface, ids)
    return ids

def gen_get_all_recursive(surface:Surface, all_properties:Dict[Surface, Dict]=None, ids:Mapping[Surface, int]=None) -> str:
    if all_properties is None:
        all_properties = all_properties
    if ids is None:
        ids = add_ids(surface)
    sdb_glsl = gen_getSDB_recursive(surface, ids) + gen_getColor_recursive(surface, ids, all_properties)
    return sdb_glsl