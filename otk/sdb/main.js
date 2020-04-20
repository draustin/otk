"use strict";

var gl;
var canvas;
var buffer;

var trace_program;
var clip_to_eye = mat4.create();

var mouse_press_event = null;
var mouse_press_eye_to_world;
// var mouse_press_eye;
// var mouse_press_ndc;

window.onload = init;

function init() {
    canvas        = document.getElementById('glscreen');
    // https://developer.mozilla.org/en-US/docs/Web/API/HTMLCanvasElement/getContext
    // preserve for picking depth
    gl            = canvas.getContext('webgl2', {depth: true, preserveDrawingBuffer: true});
    canvas.width  = 640;
    canvas.height = 480;
    
    // https://www.cs.colostate.edu/~anderson/newsite/javascript-zoom.html
    canvas.addEventListener('mousedown', handleMouseDown, false);
    canvas.addEventListener('mousemove', handleMouseMove, false);
    canvas.addEventListener('mouseup', handleMouseUp, false);
    //canvas.addEventListener("mousewheel", handleMouseWheel, false); // mousewheel duplicates dblclick function
    //canvas.addEventListener("DOMMouseScroll", handleMouseWheel, false); // for Firefox

    gl.viewport(0, 0, gl.drawingBufferWidth, gl.drawingBufferHeight);

    trace_program = new SphereTraceProgram(sdb_glsl);

    render();
}

// Port of ShereTraceRender.paintGL.
function render() {

    //window.requestAnimationFrame(render, canvas);

    var height = gl.drawingBufferHeight;
    var width = gl.drawingBufferWidth;
    var eye_to_clip = projection.eye_to_clip(height/width);
    mat4.invert(clip_to_eye, eye_to_clip);
    var clip_to_world = mat4.create()
    mat4.mul(clip_to_world, clip_to_eye, eye_to_world);
    var world_to_clip = mat4.create()
    mat4.invert(world_to_clip, clip_to_world);

    gl.depthMask(gl.TRUE);
    gl.enable(gl.DEPTH_TEST);
    gl.depthFunc(gl.ALWAYS);
    trace_program.draw(eye_to_world, eye_to_clip, [width, height], max_steps, epsilon, background_color);
    gl.depthFunc(gl.LESS);
    // TODO ray program
    // TODO wire frame program
}

// function event_to_ndc(event) {
//     console.log(event);
//     var x_window = event.clientX;
//     var  y_window = event.clientY;
//     var viewport = gl.getParameter(gl.VIEWPORT);
//     var depth_range = gl.getParameter(gl.DEPTH_RANGE);
//     var zs = new Float32Array(1);
//     //var z_window = gl.readPixels(x_window, y_window, 1, 1, gl.DEPTH_COMPONENT, gl.FLOAT, zs);
//     console.log(zs);
//     // https://www.khronos.org/opengl/wiki/Compute_eye_space_from_window_space
//     var ndc = vec4.fromValues((2*(x_window - viewport[0])/viewport[2] - 1, 2*(y_window - viewport[1])/viewport[3] - 1,
//         (2*z_window - (depth_range[1] + depth_range[0]))/(depth_range[1] - depth_range[0]), 1))
//     return ndc;
// }

// function ndc_to_eye(ndc) {
//     var eyep = vec4.create();
//     mulVecMat4(eyep, ndc, clip_to_eye);
//     var eye =  vec4.create();   
//     eye = vec4.multiplyScalar(eye, eyep, 1/eyep[3]);
//     return eye;
// }

function handleMouseDown(event) {
    if (mouse_press_event == null) {
        mouse_press_event = event;
        mouse_press_eye_to_world = mat4.clone(eye_to_world);
        // mouse_press_ndc = event_to_ndc(event);
        // mouse_press_eye = ndc_to_eye(mose_press_ndc);
    }
}

function handleMouseMove(event) {
    if (mouse_press_event != null) {
        var dx_p = event.clientX - mouse_press_event.clientX;
        var dy_p = event.clientY - mouse_press_event.clientY;
        var viewport = gl.getParameter(gl.VIEWPORT);
        var dx_ndc = 2*(dx_p - viewport[0])/viewport[2];
        var dy_ndc = 2*(dy_p - viewport[1])/viewport[3];

        if (mouse_press_event.which == 1) {
            var ndc = event_to_ndc(event);
            ndc[2] = mouse_press_ndc[2];
            eye = ndc_to_eye(ndc);
            delta = eye - mouse_press_eye;
            transform = mat4.make_translation(-delta[0], -delta[1], 0.);
            mat4.mul(eye_to_world, transform, mouse_press_eye_to_world);
            window.requestAnimationFrame(render, canvas);
        }
    }
}

function handleMouseUp(event) {
    mouse_press_event = null;
}