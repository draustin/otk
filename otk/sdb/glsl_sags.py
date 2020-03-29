from functools import singledispatch
from textwrap import dedent
import numpy as np
from . import *

__all__ = ['gen_getSag']

@singledispatch
def gen_getSag(sagfun, id:str):
    raise NotImplementedError()

@gen_getSag.register
def _(s:ZemaxConicSagFunction, id):
    string = dedent(f"""\
        float getSag{id}(in vec2 x) {{
            float rho = min(length(x.xy), {s.radius});
            float z;\n""")
    if np.isfinite(s.roc):
        string += f'    z = rho*rho/({s.roc}*(1 + sqrt(1 - {s.kappa}*rho*rho/{s.roc**2})));\n'
    else:
        string += '     z = 0.;'
    if len(s.alphas) > 0:
        string += f'    float h = {s.alphas[-1]};\n'
        string += ''.join(f'    h = h*rho + {alpha};\n' for alpha in s.alphas[-2::-1])
        string += '    z += h*rho*rho;'
    string += '    return z;\n}\n\n'
    return string

@gen_getSag.register
def _(s:SinusoidSagFunction, id):
    return dedent(f"""\
        float getSag{id}(in vec2 x) {{
            return {s.amplitude}*cos(x.x*{s.vector[0]} + x.y*{s.vector[1]});
        }}\n\n""")

@gen_getSag.register
def _(s:RectangularArraySagFunction, id):
    id_unit = id + 'Unit'
    return gen_getSag(s.unit, id_unit) + dedent(f"""\
        float getSag{id}(in vec2 x) {{
            vec2 q = mod(x + {gen_vec2(s.pitch/2)}, {gen_vec2(s.pitch)}) - {gen_vec2(s.pitch/2)};
            return getSag{id_unit}(q);
        }}\n\n""")