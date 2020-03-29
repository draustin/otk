from functools import singledispatch

@singledispatch
def f(x):
    raise NotImplementedError()

@f.register
def _(x:int):
    print(x)

@f.register
def _(x:float, y):
    print(x+y)

f(5)

f(5.4)