from .. import sdb, ri
from . import *
from . import scalar

# TODO chrome mask
def make_MLA150(small:bool=False):
    # https://www.thorlabs.com/drawings/2387c2b71216f558-2D5B16EB-0174-2DDB-0867F5C4641A7CDC/MLA150-5C-SpecSheet.pdf
    roc = 2.54e-3
    d = 140e-6
    p = 150e-6
    t = 1.2e-3
    w = 10e-3
    h = 10e-3
    unitfn = sdb.ZemaxConicSagFunction(roc, d/2)
    fn = sdb.RectangularArraySagFunction(unitfn, (p, p))
    front = sdb.Sag(fn, 1)
    sides = sdb.InfiniteRectangularPrism(w, h)
    back = sdb.Plane((0, 0, 1), -t)
    surface = sdb.IntersectionOp((front, sides, back), sdb.Box((w/2, h/2, t/2), (0, 0, t/2)))
    element = SimpleElement(surface, UniformIsotropic(ri.fused_silica), perfect_refractor)
    return element