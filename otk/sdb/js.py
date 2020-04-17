"""Display models in web pages using WebGL."""
import os
from textwrap import dedent
from . import glsl

def gen_html(sdb_glsl: str):
    """Under construction.

    Next step is to adapt opengl.py to Javascript to enable camera movement.
    Would be nice for most of the Javascript to be in a separate file.
    """

    # TODO would be nice to only have one file... maybe GLSL preprocessor switches?
    with open(os.path.join(os.path.dirname(__file__), 'trace-300es.glsl'), 'rt') as f:
        trace_glsl = f.read()

    fragment_source = glsl.sdf_glsl + sdb_glsl + trace_glsl
    return (dedent("""\
    
    <!DOCTYPE html>

    <html>
    <body>

    <script type="text/javascript">// <![CDATA[

    var gl;
    var canvas;
    var buffer;
    
    var shaderScript;
    var shaderSource;
    var vertexShader;
    var fragmentShader;
    
    window.onload = init;
    
    function init() {
    
    canvas        = document.getElementById('glscreen');
    gl            = canvas.getContext('webgl2');
    canvas.width  = 640;
    canvas.height = 480;
    
    gl.viewport(0, 0, gl.drawingBufferWidth, gl.drawingBufferHeight);
    
    buffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
    gl.bufferData(
    gl.ARRAY_BUFFER,
    new Float32Array([
      -1.0, -1.0,
       1.0, -1.0,
      -1.0,  1.0,
      -1.0,  1.0,
       1.0, -1.0,
       1.0,  1.0]),
    gl.STATIC_DRAW
    );
    
    shaderScript = document.getElementById("2d-vertex-shader");
    shaderSource = shaderScript.text;
    vertexShader = gl.createShader(gl.VERTEX_SHADER);
    gl.shaderSource(vertexShader, shaderSource);
    gl.compileShader(vertexShader);
    var compilationLog = gl.getShaderInfoLog(vertexShader);
    console.log('Vertex shader compiler log: ' + compilationLog);
    
    shaderScript   = document.getElementById("2d-fragment-shader");
    shaderSource   = shaderScript.text;
    fragmentShader = gl.createShader(gl.FRAGMENT_SHADER);
    gl.shaderSource(fragmentShader, shaderSource);
    gl.compileShader(fragmentShader);
    var compilationLog = gl.getShaderInfoLog(fragmentShader);
    console.log('Fragment shader compiler log: ' + compilationLog);
    
    program = gl.createProgram();
    gl.attachShader(program, vertexShader);
    gl.attachShader(program, fragmentShader);
    gl.linkProgram(program);
    gl.useProgram(program);
    
    
    
    render();
    
    }
    
    function render() {
    
    window.requestAnimationFrame(render, canvas);
    
    gl.clearColor(1.0, 0.0, 0.0, 1.0);
    gl.clear(gl.COLOR_BUFFER_BIT);
    positionLocation = gl.getAttribLocation(program, "a_position");
    gl.enableVertexAttribArray(positionLocation);
    gl.vertexAttribPointer(positionLocation, 2, gl.FLOAT, false, 0, 0);
    gl.drawArrays(gl.TRIANGLES, 0, 6);
    
    }

    // ]]>
    </script>
    

    <script id="2d-vertex-shader" type="x-shader/x-vertex">#version 300 es
    in vec2 a_position;
    void main() {
        gl_Position = vec4(a_position, 0, 1);
    }
    </script>
    
    <script id="2d-fragment-shader" type="x-shader/x-fragment">#version 300 es
    
    #ifdef GL_FRAGMENT_PRECISION_HIGH
      precision highp float;
    #else
      precision mediump float;
    #endif
    precision mediump int;\n\n""") +
    fragment_source +
    dedent(f"""\
    
    </script>
    
    <canvas id="glscreen"></canvas>
    
    </body>
    </html>\n\n"""))
