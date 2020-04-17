uniform int max_steps;
uniform float epsilon;
uniform vec2 iResolution;
uniform mat4 clip_to_world;
uniform mat4 world_to_clip;

vec4 getNormal(in vec4 x) {
    const float h = 0.000001; // TODO better determination of this
    const vec3 k = vec3(1,-1,0.);
    return normalize( k.xyyz*getSDB0( x + k.xyyz*h ) +
                      k.yyxz*getSDB0( x + k.yyxz*h ) +
                      k.yxyz*getSDB0( x + k.yxyz*h ) +
                      k.xxxz*getSDB0( x + k.xxxz*h ) );
}

struct SphereTrace {
    float d; // signed distance at x
    float t; // travelled distance
    vec4 x; // stopping point
    int steps;
};

SphereTrace sphereTrace(in vec4 x0, in vec4 v, in float t_max)
{
	float t = 0.;
    vec4 x = x0;
    int steps;
    float d;
    for (steps = 0; steps <= max_steps; steps++) {
		d = getSDB0(x);
        if ((d < epsilon) || (t > t_max))
            break;
        t += d;
        x = x0 + t*v;
	}
    return SphereTrace(d, t, x, steps);
}

uniform vec3 light_direction;
uniform vec4 background_color;

out vec4 FragColor;

void main()
{
    vec2 ndc = 2.0*gl_FragCoord.xy/iResolution.xy - 1.0;

    vec4 x0_world;
    vec4 v_world;
    float t_max;
    // Write x0 and v in eye coordinates
    ndc2ray(ndc, clip_to_world, x0_world, v_world, t_max);

    SphereTrace trace = sphereTrace(x0_world, v_world, t_max);

    if (trace.d < epsilon)
    {
        vec3 surface_color = getColor0(trace.x);
        vec4 normal_world = getNormal(trace.x);
        FragColor = vec4((max(dot(normal_world.xyz, light_direction), 0.) + 0.1)*surface_color, 1.);
        vec4 x_clip = trace.x*world_to_clip;
        // gl_FragDepth is in window-space coordinates, so its range is gl_DepthRange.near to gl_DepthRange.far.
        gl_FragDepth = (x_clip.z/x_clip.w*gl_DepthRange.diff + gl_DepthRange.near + gl_DepthRange.far)/2.;
    }
    else {
        FragColor = background_color;
        gl_FragDepth = gl_DepthRange.far;
    }
//    float ss = float(trace.steps)/max_steps;
//    gl_FragColor = vec4(ss, ss, ss, 1.0);
}