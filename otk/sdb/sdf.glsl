float sphereSDF(in vec3 o, in float r, in vec3 x)
{
    return length(x - o) - r;
}

float circleSDF(in vec2 o, in float r, in vec2 x) {
    return length(x - o) - r;
}

float rectangleSDF(in vec2 center, in vec2 half_size, in vec2 x) {
    vec2 q = abs(x - center) - half_size;
    // https://www.iquilezles.org/www/articles/distfunctions/distfunctions.htm
    return length(max(q, 0.0)) + min(max(q.x, q.y), 0.0);
}

float SphericalSagSDB(in float z_vertex, in float roc, in vec3 x) {
    float z_center = z_vertex + roc;
    return min(sphereSDF(vec3(0, 0, z_center), abs(roc), x), x.z - z_center);
}

void ndc2ray(in vec2 ndc, in mat4 invP, out vec4 w0, out vec4 v, out float t_max)
{
    vec4 n0 = vec4(ndc, -1., 1.);
    vec4 nf = vec4(ndc, 1., 1.);

    vec4 w0p = n0*invP;
    w0 = w0p / w0p.w;
    vec4 wfp = nf*invP;
    vec4 wf = wfp/wfp.w;

    vec4 vp = wf - w0;
    t_max = length(vp);
    v = vp/t_max;
}

