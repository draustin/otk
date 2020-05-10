"""Miscellaneous non-optics, non-math stuff."""
import os

ROOT_DIR = os.path.dirname(__file__)

class Delegate:
    # Inspired by https://gist.github.com/dubslow/b8996308fc6af2437bef436fa28e86fa.
    def __init__(self, field: str, subfield: str):
        self.field = field
        self.subfield = subfield

    def __get__(self, instance, cls):
        return getattr(getattr(instance, self.field), self.subfield)

    def __set__(self, instance, value):
        setattr(getattr(instance, self.field), self.subfield, value)



if __name__ == '__main__':
    class X:
        def __init__(self, a):
            self.a = a

        def add(self, b):
            return self.a + b


    class Y:
        def __init__(self, a):
            self.x = X(a)

        a = Delegate('x', 'a')
        add = Delegate('x', 'add')

    y = Y(1)
    print(y.a)
    print(y.add(2))
    y.a = 3
    print(y.a)
    print(y.add(2))






