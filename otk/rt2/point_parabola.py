from sympy import symbols, S, simplify

x0, y0, z0, lamb = symbols('x0 y0 z0 lamb')

a = S(1)/2
b = -(z0 + 1)
c = (S(1)/2 + 2*z0)
d = x0**2 + y0**2 - z0

discr = simplify(18*a*b*c*d - 4*b**3*d + b**2*c**2 - 4*a*c**3 - 27*a**2*d**2)