"use strict";

function link_program(vertexSource, fragmentSource) {
    var log;

    var vertexShader = gl.createShader(gl.VERTEX_SHADER);
    gl.shaderSource(vertexShader, vertexSource);
    gl.compileShader(vertexShader);
    log = gl.getShaderInfoLog(vertexShader);
    console.log('Vertex shader compiler log: ' + log);

    //console.log('fragmentSource = \n' + fragmentSource);
    var fragmentShader = gl.createShader(gl.FRAGMENT_SHADER);
    gl.shaderSource(fragmentShader, fragmentSource);
    gl.compileShader(fragmentShader);
    log = gl.getShaderInfoLog(fragmentShader);
    console.log('Fragment shader compiler log: ' + log);

    var program = gl.createProgram();
    gl.attachShader(program, vertexShader);
    gl.attachShader(program, fragmentShader);
    gl.linkProgram(program);
    log = gl.getProgramInfoLog(program);
    console.log('Program link log: ' + log);

    return program;
}

function SphereTraceProgram(sdb_glsl) {
    var fragment_source = `#version 300 es

#ifdef GL_FRAGMENT_PRECISION_HIGH
    precision highp float;
#else
    precision mediump float;
#endif
precision mediump int;
`
        + sdf_glsl + sdb_glsl + trace_glsl;

    var program = link_program('#version 300 es\n\n' + trace_vertex_source, fragment_source)
    var position_buffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, position_buffer);
    gl.bufferData(
        gl.ARRAY_BUFFER,
        new Float32Array([
        -1.0, -1.0,
        -1.0, 1.0,
        1.0,  -1.0,
        1.0,  1.0]),
        gl.STATIC_DRAW
    );
    this.program = program;
    this.position_buffer = position_buffer;
}

SphereTraceProgram.prototype.draw = function(eye_to_world, eye_to_clip, resolution, max_steps, epsilon, background_color) {
    gl.bindBuffer(gl.ARRAY_BUFFER, this.position_buffer);
    var loc = gl.getAttribLocation(this.program, "position");
    gl.enableVertexAttribArray(loc);
    gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);

    var clip_to_world = mat4.create();
    var world_to_clip = mat4.create();
    var clip_to_eye = mat4.create();
    mat4.invert(clip_to_eye, eye_to_clip);
    mat4.mul(clip_to_world, clip_to_eye, eye_to_world);
    mat4.invert(world_to_clip, clip_to_world);

    gl.useProgram(this.program);
    gl.uniform2f(gl.getUniformLocation(this.program, 'iResolution'), resolution[0], resolution[1]);
    gl.uniformMatrix4fv(gl.getUniformLocation(this.program, 'clip_to_world'), false, clip_to_world);
    gl.uniformMatrix4fv(gl.getUniformLocation(this.program, 'world_to_clip'), false, world_to_clip);
    // glMatrix.mat4 is in column-major order. Light to be coming from behind eye, in world coordinates. So want 3rd row.
    gl.uniform3f(gl.getUniformLocation(this.program, 'light_direction'), eye_to_world[2], eye_to_world[6], eye_to_world[10]);
    gl.uniform4f(gl.getUniformLocation(this.program, 'background_color'), background_color[0], background_color[1], background_color[2], background_color[3]);
    gl.uniform1i(gl.getUniformLocation(this.program, 'max_steps'), max_steps);
    gl.uniform1f(gl.getUniformLocation(this.program, 'epsilon'), epsilon);
    
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
}

// function orthographic(l, r, b, t, n, f) {
//     return mat4.fromValues(2*(r - l), 0, 0, 0, 0, 2/(t -b), 0, 0, 0, 0, -2/(f - n), 0, -(r + l)/(r - l), -(t + b)/(t - b), -(f + n)/(f - n), 1); 
// }

function Orthographic(half_width, z_far) {
    this.half_width = half_width;
    this.z_far = z_far;
}

Orthographic.prototype.eye_to_clip = function(aspect) {
    var half_height = this.half_width*aspect;
    var m = mat4.create();
    return mat4.transpose(m, mat4.ortho(m, -this.half_width, this.half_width, -half_height, half_height, 0., this.z_far));
}

function Perspective(fov, z_near, z_far) {
    this.fov = fov;
    this.z_near = z_near;
    this.z_far = z_far;
}

Perspective.prototype.eye_to_clip = function(aspect) {
    var half_width = this.z_near*Math.tan(this.fov/2);
    var half_height = half_width*aspect;
    var m = mat4.create();
    return mat4.transpose(m, mat4.frustum(m, -half_width, half_width, -half_height, half_height, this.z_near, this.z_far));
}