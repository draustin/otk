## Obtaining z of a clicked pixel.

Turns out this is a ginormous pain.

Unlike OpenGL's readPixels, [WebGL's readpixels](https://developer.mozilla.org/en-US/docs/Web/API/WebGLRenderingContext/readPixels) can only read from the color buffer.

### Idea 1

Write fragment z to an additional output.

Tried this:

    const x_clip_render_buffer = gl.createRenderbuffer();
    gl.bindRenderbuffer(gl.RENDERBUFFER, x_clip_render_buffer);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.framebufferRenderbuffer(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT1, gl.RENDERBUFFER, x_clip_render_buffer);

But `WebGL: INVALID_OPERATION: framebufferRenderbuffer: no framebuffer bound`. Looks like you [can't attach color attachments to the default framebuffer](https://community.khronos.org/t/color-attachment-to-default-framebuffer/66029).

In Chrome on Macbook Pro,

    console.log(gl.getParameter(gl.IMPLEMENTATION_COLOR_READ_FORMAT).toString(16));
    console.log(gl.getParameter(gl.IMPLEMENTATION_COLOR_READ_TYPE).toString(16));

gives 1908 and 1401. [Table of WebGL constants](https://developer.mozilla.org/en-US/docs/Web/API/WebGL_API/Constants) says these are RGBA and UNSIGNED_BYTE. Not much help.

### Idea 2

When the user clicks, render one pixel with the fragment shader set to a mode that encodes the z value in the color.

From reading WebGL/OpenGL docs, gather that I need to 

1. Make a new Framebuffer object and attach a Renderbuffer of size (1,1) to its color attachment 0.

Can set Renderbuffer size with [`renderbufferStorage`](https://developer.mozilla.org/en-US/docs/Web/API/WebGLRenderingContext/renderbufferStorage) method.

If I understand ['viewport'] correctly: with the 1 pixel framebuffer our pixel will have window coordinates (0, 0) or maybe (0.5 0.5). If the clicked coordinates on the displayed framebuffer are `(x, y)`, then calling `viewport(-x, -y, width, height)` should make the same normalized device coordinates map to (0, 0).