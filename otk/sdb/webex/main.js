"use strict";

var gl;
var canvas;
var buffer;

var trace_program;
var ray_program;
var clip_to_eye = mat4.create();

var mouse_press_event = null;
var mouse_press_eye_to_world;

var pick_framebffer;
var mouse_press_eye;
var mouse_press_ndc;

window.onload = init;

function init() {
    canvas        = document.getElementById('glscreen');
    // https://developer.mozilla.org/en-US/docs/Web/API/HTMLCanvasElement/getContext
    // TODO preserve for picking depth??
    gl            = canvas.getContext('webgl2', {depth: true, preserveDrawingBuffer: true});
    // canvas.width  = 640;
    // canvas.height = 480;

    canvas.oncontextmenu = function(e) {
        e.preventDefault();
        return false;
    }
    
    // https://www.cs.colostate.edu/~anderson/newsite/javascript-zoom.html
    canvas.addEventListener('mousedown', handleMouseDown, false);
    canvas.addEventListener('mousemove', handleMouseMove, false);
    canvas.addEventListener('mouseup', handleMouseUp, false);
    canvas.addEventListener("wheel", handleMouseWheel, false); // mousewheel duplicates dblclick function
    //canvas.addEventListener("DOMMouseScroll", handleMouseWheel, false); // for Firefox

    trace_program = new SphereTraceProgram(sdb_glsl);
    ray_program = new RayProgram();
    ray_program.set_rays(rays);

    pick_framebffer = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, pick_framebffer);
    const pick_renderbuffer = gl.createRenderbuffer();
    gl.bindRenderbuffer(gl.RENDERBUFFER, pick_renderbuffer);
    gl.renderbufferStorage(gl.RENDERBUFFER, gl.RGBA4, 1, 1);
    gl.framebufferRenderbuffer(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.RENDERBUFFER, pick_renderbuffer);

    gl.bindFramebuffer(gl.FRAMEBUFFER, null);

    render();
    //resizeCanvas();
}

function resizeCanvas() {
    var width = canvas.clientWidth;
    var height = canvas.clientHeight;
    if (canvas.width != width ||
        canvas.height != height) {
      canvas.width = width;
      canvas.height = height;
  }
}

// Port of ShereTraceRender.paintGL.
function render() {
    resizeCanvas();
    gl.viewport(0, 0, gl.drawingBufferWidth, gl.drawingBufferHeight);

    //window.requestAnimationFrame(render, canvas);
    render_(false);
}

function render_(depth_out) {
    var height = gl.drawingBufferHeight;
    var width = gl.drawingBufferWidth;
    var eye_to_clip = projection.eye_to_clip(height/width);
    mat4.invert(clip_to_eye, eye_to_clip);
    var clip_to_world = mat4.create()
    mat4.mul(clip_to_world, clip_to_eye, eye_to_world);
    var world_to_clip = mat4.create()
    mat4.invert(world_to_clip, clip_to_world);

    const viewport = gl.getParameter(gl.VIEWPORT);

    gl.depthMask(true);
    gl.enable(gl.DEPTH_TEST);
    gl.depthFunc(gl.ALWAYS);
    trace_program.draw(eye_to_world, eye_to_clip, viewport, max_steps, epsilon, background_color, depth_out);
    gl.depthFunc(gl.LESS);
    ray_program.draw(world_to_clip);
    // TODO wire frame program
}

function event_to_ndc(event) {
    const x_window = event.clientX;
    const y_window = gl.drawingBufferHeight - event.clientY - 1;
    const viewport = gl.getParameter(gl.VIEWPORT);
    //const depth_range = gl.getParameter(gl.DEPTH_RANGE);

    gl.bindFramebuffer(gl.FRAMEBUFFER, pick_framebffer);
    gl.viewport(-x_window, -y_window, gl.drawingBufferWidth, gl.drawingBufferHeight);
    render_(true);
    let packed = new Uint8Array(4);
    //packed[3] = 1;
    //packed[2] = 3;
    gl.readPixels(0, 0, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, packed);
    const z0to1 = unpack(packed);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, gl.drawingBufferWidth, gl.drawingBufferHeight);

    // https://www.khronos.org/opengl/wiki/Compute_eye_space_from_window_space
    const x_ndc = 2*(x_window - viewport[0])/viewport[2] - 1;
    const y_ndc = 2*(y_window - viewport[1])/viewport[3] - 1;
    const z_ndc = z0to1*2 - 1;
    var ndc = vec4.fromValues(x_ndc, y_ndc, z_ndc, 1);
    return ndc;
}

function ndc_to_eye(ndc) {
    let eyep = vec4.create();
    mulVecMat4(eyep, ndc, clip_to_eye);
    let eye =  vec4.create();   
    eye = vec4.scale(eye, eyep, 1/eyep[3]);
    return eye;
}

function unpack(packed) {
    const factors = [1/256, 1/(256*256), 1/(256*256*256), 1/(256*256*256*256)];
    let v = 0;
    for (let i = 0; i < 4; i++) {
        v += packed[i]*factors[i];
    }
    return v;
}

function handleMouseDown(event) {
    if (mouse_press_event == null) {
        mouse_press_event = event;
        mouse_press_eye_to_world = mat4.clone(eye_to_world);
        mouse_press_ndc = event_to_ndc(event);
        mouse_press_eye = ndc_to_eye(mouse_press_ndc);
    }
}

function handleMouseMove(event) {
    if (mouse_press_event != null) {
        // https://developer.mozilla.org/en-US/docs/Web/API/MouseEvent/button
        if (mouse_press_event.button == 0 && !mouse_press_event.ctrlKey) { // "main" button, usually left
            let ndc = event_to_ndc(event);
            ndc[2] = mouse_press_ndc[2];
            const eye = ndc_to_eye(ndc);
            let delta_eye = vec4.create();
            vec4.subtract(delta_eye, eye, mouse_press_eye);
            const transform = make_translation(-delta_eye[0], -delta_eye[1], -delta_eye[2]);
            mat4.mul(eye_to_world, transform, mouse_press_eye_to_world);
        } else if (mouse_press_event.button == 2 || (mouse_press_event.button == 0 && mouse_press_event.ctrlKey)) { 
            // secondary button, usually right, or left & control (for Mac Touchpad)
            const viewport = gl.getParameter(gl.VIEWPORT);
            const dx_ndc = -2*(event.screenX - mouse_press_event.screenX)/viewport[2];
            const dy_ndc = -2*(event.screenY - mouse_press_event.screenY)/viewport[3];
            const phi = dx_ndc*4*Math.PI;
            const theta = dy_ndc*4*Math.PI;
            let transform = make_translation(-mouse_press_eye[0], -mouse_press_eye[1], -mouse_press_eye[2]);
            mat4.mul(transform, transform, make_y_rotation(phi));
            mat4.mul(transform, transform, make_x_rotation(theta));
            mat4.mul(transform, transform, make_translation(mouse_press_eye[0], mouse_press_eye[1], mouse_press_eye[2]));
            mat4.mul(eye_to_world, transform, mouse_press_eye_to_world);
        }
        window.requestAnimationFrame(render, canvas);
    }
}

function handleMouseUp(event) {
    mouse_press_event = null;
}

function handleMouseWheel(event) {
    var by = 1.5;
    var factor;
    if (event.deltaY > 0)
        factor = by;
    else if (event.deltaY < 0)
        factor = 1/by;
    projection = projection.zoom(factor);
    window.requestAnimationFrame(render, canvas);
}