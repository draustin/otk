import numpy as np
from glumpy import app, gl, glm, gloo
from otk.rt2 import *

vertex = """
attribute vec2 position;
void main (void)
{
    gl_Position = vec4(position, 0.0, 1.0);
}
"""

fragment = """
uniform float iTime;
uniform vec2 iResolution;
uniform mat4 invP;

struct DID
{
    float d;
    int id;
};

const int max_steps = 1000;
const float epsilon = 0.0001;

DID unionSDB(in DID a, in DID b)
{
    // Ternary operator illegal on structs.
    // See https://www.reddit.com/r/glsl/comments/ea7o4i/weird_ternary_operator_and_struct_combination/
    if (a.d <= b.d)
        return a;
    else
        return b;
}

DID edgeUnionSDB(in DID a, in DID b, in float w, in int eid) {
	float d = min(a.d, b.d);
    int id;
    float z = a.d - b.d;
    if (z <= - w)
       id = a.id;
    else if (z >= w)
        id = b.id;
  	else
        id = eid;
    return DID(d, id);
}

DID intersectionSDB(in DID a, in DID b) {
    if (a.d >= b.d)
        return a;
   	else
        return b;
}

DID differenceSDB(in DID a, in DID b) {
    if (a.d >= -b.d)
        return a;
    else
        return DID(-b.d, b.id);
}

DID edgeDifferenceSDB(in DID a, in DID b, in float w, in int eid) {
	float d = max(a.d, -b.d);
    int id;
    float z = a.d + b.d;
    if (z >= w)
       id = a.id;
    else if (z <= -w)
        id = b.id;
  	else
        id = eid;
    return DID(d, id);
}


float sphereSDF(in vec3 o, in float r, in vec3 x)
{
    return length(x - o) - r;
}

float cylinderSDF(in vec2 o, in float r, in vec3 x) {
    return length(x.xy - o) - r;
}

DID getSDB0(in vec4 x)
{
    return DID(sphereSDF(vec3(0., 0., 0.5), 0.7, x.xyz), 0);
}

DID getSDB1(in vec4 x)
{
    return DID(sphereSDF(vec3(0., 0., -0.5), 0.8, x.xyz), 1);
}

DID getSDB2(in vec4 x) {
    return DID(cylinderSDF(vec2(0.1, 0.), 0.2, x.xyz), 2);
}

mat4 rotation(in vec3 v, in float theta) {
    float cs = cos(theta);
    float sn = sin(theta);

    return mat4(vec4(cs + v.x*v.x*(1. - cs), v.x*v.y*(1. - cs) + v.z*sn, v.x*v.z*(1. - cs) - v.y*sn, 0.),
                vec4(v.x*v.y*(1. - cs) - v.z*sn, cs + v.y*v.y*(1. - cs), v.y*v.z*(1. - cs) + v.x*sn, 0.),
                vec4(v.x*v.z*(1. - cs) + v.y*sn, v.y*v.z*(1. - cs) - v.x*sn, cs + v.z*v.z*(1. - cs), 0.),
                vec4(0., 0., 0., 1.));
}


DID getSDB(in vec4 x)
{
    float theta = iTime;
    mat4 S = rotation(normalize(vec3(1., 1., 0.)), theta);
    x = S*x;
    float edge_width = 0.03;
    return edgeDifferenceSDB(edgeUnionSDB(getSDB0(x), getSDB1(x), edge_width, 3), getSDB2(x), edge_width, 3);
}

vec4 getNormal(in vec4 x) {
    const float h = 0.0001; // or some other value
    const vec3 k = vec3(1,-1,0.);
    return normalize( k.xyyz*getSDB( x + k.xyyz*h ).d +
                      k.yyxz*getSDB( x + k.yyxz*h ).d +
                      k.yxyz*getSDB( x + k.yxyz*h ).d +
                      k.xxxz*getSDB( x + k.xxxz*h ).d );
}


DID sphereTrace(in vec4 x0, in vec4 v, in float t_max)
{
	float t = 0.;
    int iid = -1;
    vec4 x = x0;
    for (int steps = 0; steps <= max_steps; steps++) {
		DID did = getSDB(x);
        if (did.d < epsilon)
        {
            iid = did.id;
            break;
        }
        t += did.d;
        x = x0 + t*v;
	}
    return DID(t, iid);
}

void ndc2ray(in vec2 ndc, in mat4 invP, out vec4 w0, out vec4 v, out float t_max)
{
    vec4 n0 = vec4(ndc, -1., 1.);
    vec4 nf = vec4(ndc, 1., 1.);

    vec4 w0p = invP*n0;
    w0 = w0p / w0p.w;
    vec4 wfp = invP*nf;
    vec4 wf = wfp/wfp.w;

    vec4 vp = wf - w0;
    t_max = length(vp);
    v = vp/t_max;
}

mat4 orthographic(in float l, in float r, in float b, in float t, in float n, in float f)
{
    return mat4(vec4(2./(r-l), 0., 0., 0.),
                vec4(0., 2./(t-b), 0., 0.),
                vec4(0., 0., 2./(n-f), 0.),
                vec4((r+l)/(l-r), (t+b)/(b-t), (f+n)/(n-f), 1));
}


// same as Optics.jl, but untested.
mat4 perspective(in float l, in float r, in float b, in float t, in float n, in float f)
{
    return mat4(vec4(2.*n/(r-l), 0., 0., 0.),
                vec4(0., 2.*n/(t-b), 0., 0.),
                vec4((r+l)/(r - l), (t+b)/(t-b), (f+n)/(f-n), -1.),
                vec4(0., 0., 2.*f*n/(f-n), 0.));
}

mat4 lookat(in vec3 eye, in vec3 center, in vec3 y) {
    vec3 z = normalize(eye - center);
    vec3 x = normalize(cross(y, z));
    y = cross(z, x);
    return mat4(vec4(x, 0.), vec4(y, 0.), vec4(z, 0.), vec4(eye, 1.));
}

const vec3 colors[4] = vec3[4](vec3(1., 0., 0.), vec3(0., 1., 0.), vec3(0., 0., 1.0), vec3(0., 0., 0.));

void main()
{
    vec2 ndc = 2.0*gl_FragCoord.xy/iResolution.xy - 1.0;

    vec4 x0;
    vec4 v;
    float t_max;
    ndc2ray(ndc, invP, x0, v, t_max);

    DID did = sphereTrace(x0, v, t_max);
    float t = did.d;
    int id = did.id;

    if (id >= 0)
    {
        vec4 x = x0 + v*t;
        vec3 color = colors[id];
        vec4 normal = getNormal(x);
        gl_FragColor = vec4(max(dot(normal, vec4(0., 0., 1., 0.)), 0.)*color, 1.);
    }
    else
        gl_FragColor = vec4(1., 0., 1., 1.0);
}
"""

world_to_camera = np.linalg.inv(lookat([0, 0.0, 5.0], [0.0, 0.0, 0.0]))
window = app.Window(width=800, height=800)

@window.event
def on_draw(dt):
    window.clear()
    program.draw(gl.GL_TRIANGLE_STRIP)
    program["iTime"] += dt

@window.event
def on_resize(width, height):
    half_width = 2
    half_height = half_width*height/width
    projection = orthographic(-half_width, half_width, -half_height, half_height, 0.1, 20)
    program["invP"] = np.linalg.inv(world_to_camera @ projection)
    program["iResolution"] = width, height


# @window.event
# def on_mouse_scroll(x, y, dx, dy):
#     scale = program["scale"]
#     program["scale"] = min(max(1, scale + .01 * dy * scale), 100)

program = gloo.Program(vertex, fragment, count=4)
program['position'] = [(-1,-1), (-1,1), (1,-1), (1,1)]
program["iTime"] = 0
# program["scale"] = 10
app.run(framerate=60)