"""Check that Numba supports closures."""
import numpy as np
import numba

def make_f(y, jit):
    def f(x):
        return x + y
    if jit:
        f = numba.njit(f)
    return f

class SimpleThing:
    def __init__(self, y):
        self.y = y

    def make_f(self, jit):
        y = self.y
        def f(x):
            return x + y
        if jit:
            f = numba.njit(f)
        return f

class CompoundThing:
    def __init__(self, child_thing):
        self.child_thing = child_thing

    def make_f(self, jit):
        child_f = self.child_thing.make_f(jit)
        def f(x):
            return x + child_f(x)
        if jit:
            f = numba.njit(f)
        return f

f = make_f(1, True)
print(f(1))

simple_thing = SimpleThing(1)
compound_thing = CompoundThing(simple_thing)

f = compound_thing.make_f(True)
print(f(1))

@numba.njit
def sumit(x):
    return np.sum(x)

sumit(np.asarray([1,2,3]))

class Interval:
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def make_f(self):
        a = self.a
        @numba.njit()
        def f(x):
            return a + x
        return f

f = Interval(1,2).make_f()
print(f(4))