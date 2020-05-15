"""Miscellaneous non-optics, non-math stuff."""
import os
import yaml

DESIGNS_DIR = os.path.join(os.path.dirname(__file__), 'designs')
PROPERTIES_DIR = os.path.join(os.path.dirname(__file__), 'properties')

def load_config() -> dict:
    for path in os.curdir, os.path.expanduser('~'):
        try:
            with open(os.path.join(path, 'otk.yml'), 'rt') as file:
                return yaml.load(file, Loader=yaml.FullLoader)
        except FileNotFoundError:
            pass
    return {}


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






