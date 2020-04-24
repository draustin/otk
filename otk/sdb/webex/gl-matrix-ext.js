"use strict";

const { mat4, mat3, vec4, vec3, vec2 } = glMatrix;

function mulVecMat4(out, v, m) {
    let x = v[0], y = v[1], z = v[2], w = v[3];
    out[0] = m[0]*x + m[1]*y + m[2]*z + m[3]*w;
    out[1] = m[4]*x + m[5]*y + m[6]*z + m[7]*w;
    out[2] = m[8]*x + m[9]*y + m[10]*z + m[11]*w;
    out[3] = m[12]*x + m[13]*y + m[14]*z + m[15]*w;
    return out;
}

function make_translation(x, y, z) {
    return mat4.fromValues(1., 0., 0., x, 0., 1., 0., y, 0., 0., 1., z, 0., 0., 0., 1);
}

function make_x_rotation(rad) {
    const s = Math.sin(rad);
    const c = Math.cos(rad);
    return mat4.fromValues(1., 0., 0., 0., 0., c, -s, 0., 0., s, c, 0., 0., 0., 0., 1.);
}

function make_y_rotation(rad) {
    const s = Math.sin(rad);
    const c = Math.cos(rad);
    return mat4.fromValues(c, 0., s, 0., 0., 1., 0., 0., -s, 0., c, 0., 0., 0., 0., 1.);
}


