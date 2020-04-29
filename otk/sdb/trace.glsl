uniform int max_steps;
uniform float epsilon;
uniform ivec4 viewport;
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
uniform bool depth_out;

#if __VERSION__ == 300
    out vec4 FragColor;
#endif

const vec4 bitSh = vec4(1., 256., 256. * 256., 256. * 256. * 256.);
const vec4 bitMsk = vec4(vec3(1./256.0), 0.);

vec4 pack_0to1 (float value) {
    if (value >= 1.)
        return vec4(1., 1., 1., 1.);
    else if (value <= 0.)
        return vec4(0., 0., 0., 0.);
    else {
        vec4 comp = fract(value * bitSh);
        comp -= comp.yzww * bitMsk;
        return comp;
    }
}

// https://stackoverflow.com/questions/34963366/encode-floating-point-data-in-a-rgba-texture
vec4 pack(float value, float rangeMin, float rangeMax) {
    float zeroToOne = (value - rangeMin) / (rangeMax - rangeMin);
    return pack_0to1(value);
}

void main()
{
    vec2 ndc = 2.0*(gl_FragCoord.xy - vec2(viewport.xy))/vec2(viewport.zw) - 1.0;

    vec4 x0_world;
    vec4 v_world;
    float t_max;
    // Write x0 and v in eye coordinates
    ndc2ray(ndc, clip_to_world, x0_world, v_world, t_max);

    SphereTrace trace = sphereTrace(x0_world, v_world, t_max);
    vec4 x_clip = trace.x*world_to_clip;
    
    if (trace.d < epsilon) {
        // gl_FragDepth is in window-space coordinates, so its range is gl_DepthRange.near to gl_DepthRange.far.
        gl_FragDepth = (x_clip.z/x_clip.w*gl_DepthRange.diff + gl_DepthRange.near + gl_DepthRange.far)/2.;
    } else {
        gl_FragDepth = gl_DepthRange.far;
    }

    vec4 color;

    if (depth_out) {
        color = pack(gl_FragDepth, gl_DepthRange.near, gl_DepthRange.far);
    } else {
        if (trace.d < epsilon)
        {
            vec3 surface_color = getColor0(trace.x);
            vec4 normal_world = getNormal(trace.x);
            color = vec4((max(dot(normal_world.xyz, light_direction), 0.) + 0.1)*surface_color, 1.);
            
        }
        else {
            color = background_color;
        }
    }
    #if __VERSION__ == 300
        FragColor
    #else
        gl_FragColor
    #endif
    = color;
}