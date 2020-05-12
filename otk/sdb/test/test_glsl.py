from textwrap import dedent
from otk import sdb

def test_primitives():
    s0 =  sdb.Sphere(0.5, (1.0, 2.0, 3.0))
    s1 = sdb.InfiniteCylinder(0.5, (1.0, 2.0))
    s2 = sdb.SphericalSag(0.6, 1, (1, 2, 3))
    s3 = sdb.Hemisphere(0.5, (1.0, 2.0, 3.0), -1)
    s4 = sdb.ZemaxConic(0.1, 0.05, 1., 0.9, [0.1, 0.2], [0.5, 0.6, 0.7])
    ids = {s0: 0, s1: 1, s2: 2, s3: 3, s4: 4}


    assert sdb.gen_getSDB(s0, ids) == dedent("""\
            float getSDB0(in vec4 x) {
                return sphereSDF(vec3(1.0, 2.0, 3.0), 0.5, x.xyz);
            }\n\n""")

    assert sdb.gen_getSDB(s1, ids) == dedent("""\
            float getSDB1(in vec4 x) {
                return circleSDF(vec2(1.0, 2.0), 0.5, x.xy);
            }\n\n""")

    assert sdb.gen_getSDB(s2, ids) == dedent("""\
            float getSDB2(in vec4 x) {
                return min(1.0*(length(x.xyz - vec3(1.0, 2.0, 3.6)) - 0.6), -1.0*(x.z - 3.6));
            }\n\n""")

    assert sdb.gen_getSDB(s3, ids) == dedent("""\
                float getSDB3(in vec4 x) {
                    return min(sphereSDF(vec3(1.0, 2.0, 3.0), 0.5, x.xyz), -1*(x.z - 3.0));
                }\n\n""")

    # Generated code contains derived numbers at high precision. Won't try to check for now.
    # assert sdb.gen_getSDB(s, {s:0}) == dedent(f"""\
    #     float getSDB0(in vec4 x) {{
    #         float rho = min(length(x.xy - vec2(0.5, 0.6)), 0.05);
    #         float z = 0.1 - 1*sqrt(
    #         return -1.0*z/{s.lipschitz};
    #     }}\n\n""")

def test_UnionOp():
    s0 = sdb.Sphere(0.5, (1.0, 2.0, 3.0))
    s1 = sdb.InfiniteCylinder(0.5, (1.0, 2.0))
    s2 = sdb.SphericalSag(0.6, 1, (1, 2, 3))
    s3 = sdb.UnionOp((s0, s1, s2))
    ids = {s0: 0, s1: 1, s2: 2, s3: 3}

    assert sdb.gen_getSDB(s3, ids) == dedent("""\
        float getSDB3(in vec4 x) {
            float dp;
            float d = getSDB0(x);
            dp = getSDB1(x);
            if (dp < d) d = dp;
            dp = getSDB2(x);
            if (dp < d) d = dp;
            return d;
        }\n\n""")

def test_IntersectionOp():
    s0 = sdb.Sphere(0.5, (1.0, 2.0, 3.0))
    s1 = sdb.InfiniteCylinder(0.5, (1.0, 2.0))
    s2 = sdb.SphericalSag(0.6, 1, (1, 2, 3))
    s3 = sdb.IntersectionOp((s0, s1, s2))
    ids = {s0: 0, s1: 1, s2: 2, s3: 3}
    assert sdb.gen_getSDB(s3, ids) == dedent("""\
        float getSDB3(in vec4 x) {
            float dp;
            float d = getSDB0(x);
            dp = getSDB1(x);
            if (dp > d) d = dp;
            dp = getSDB2(x);
            if (dp > d) d = dp;
            return d;
        }\n\n""")



def test_DifferenceOp():
    s0 = sdb.Sphere(0.5, (1.0, 2.0, 3.0))
    s1 = sdb.InfiniteCylinder(0.5, (1.0, 2.0))
    s2 = sdb.DifferenceOp(s0, s1)
    ids = {s0: 0, s1: 1, s2: 2}
    assert sdb.gen_getSDB(s2, ids) == dedent("""\
        float getSDB2(in vec4 x) {
            return max(getSDB0(x), -getSDB1(x));
        }\n\n""")


