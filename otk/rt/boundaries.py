import numpy as np

class Boundary:
    """Immutable class representing a two-dimension region."""
    def __init__(self, finite):
        self.finite = finite

    def is_inside(self, x, y):
        """

        Args:
            x:
            y:

        Returns:
            bool (numeric): Shape should that of x and y broadcasted. Value indicates whether the corresponding (x, y)
                point is inside the boundary.
        """
        raise NotImplementedError()

    def translate(self, dx, dy):
        """Move the boundary.

        Returns:
            Copy of self, moved.
        """
        raise NotImplementedError()

class InfiniteBoundary(Boundary):
    def __init__(self, display_size=1):
        Boundary.__init__(self, False)
        self.display_size = display_size

    def is_inside(self, x, y):
        return np.full(np.broadcast(x, y).shape, True)

    def __str__(self):
        return 'infinite'

    def __repr__(self):
        return 'InfiniteBoundary(display_size=%.3f mm)'%(self.display_size*1e3)

    def make_perimeter(self):
        return np.asarray([-0.5, 0.5, 0.5, -0.5, -0.5])*self.display_size, np.asarray([-0.5, -0.5, 0.5, 0.5, -0.5])*self.display_size

    def sample(self, step=None):
        if step is None:
            step = self.display_size/16
        num_points = int(np.ceil(self.display_size/step))
        r = np.linspace(-0.5, 0.5, num_points)*self.display_size
        return r[:, None], r

    def translate(self, dx, dy):
        return InfiniteBoundary(self.display_size)

class SquareBoundary(Boundary):
    """A square boundary aligned with the x-y axes.

    Attributes:
        side_length (scalar): Side length.
        center ((2, ) array): Center of square.
    """
    def __init__(self, side_length, center=(0, 0)):
        Boundary.__init__(self, True)
        self.side_length = side_length
        assert len(center) == 2
        self.center = np.asarray(center)

    def __repr__(self):
        return 'SquareBoundary(side_length=%.3f mm, center=(%.3f, %.3f) mm)'%(self.side_length*1e3, *(self.center*1e3))

    def is_inside(self, x, y):
        x = np.asarray(x)
        y = np.asarray(y)
        xc, yc = self.center
        return (abs(x - xc) <= self.side_length/2) & (abs(y - yc) <= self.side_length/2)

    def __str__(self):
        string = 'square of size %.3f mm'%(self.side_length*1e3)
        if np.array_equal(self.center, (0, 0)):
            string += 'and center (%.3f mm, %.3f mm)'%(self.center[0]*1e3, self.center[1]*1e3)
        return string

    def make_perimeter(self):
        return np.asarray([-0.5, 0.5, 0.5, -0.5, -0.5])*self.side_length, np.asarray([-0.5, -0.5, 0.5, 0.5, -0.5])*self.side_length

    def sample(self, step=None):
        if step is None:
            step = self.side_length/16
        num_points = int(np.ceil(self.side_length/step))
        r = np.linspace(-0.5, 0.5, num_points)*self.side_length
        return r[:, None] + self.center[0], r + self.center[1]

    def translate(self, dx, dy):
        return SquareBoundary(self.side_length, self.center + (dx, dy))

    def get_interval(self, origin, vector):
        ox, oy = origin
        vx, vy = vector
        dx0 = np.divide(-self.side_length/2 - ox, vx)
        dx1 = np.divide(self.side_length/2 - ox, vx)
        dy0 = np.divide(-self.side_length/2 - oy, vy)
        dy1 = np.divide(self.side_length/2 - oy, vy)
        d1 = max(min(dx0, dx1), min(dy0, dy1))
        d2 = min(max(dx0, dx1), max(dy0, dy1))
        return d1, d2

class RectangleBoundary(Boundary):
    def __init__(self, width, height, center=(0, 0)):
        Boundary.__init__(self, True)
        self.width = width
        self.height = height
        assert len(center) == 2
        self.center = center

    def is_inside(self, x, y):
        x = np.asarray(x)
        y = np.asarray(y)
        xc, yc = self.center
        return (abs(x - xc) <= self.width/2) & (abs(y - yc) <= self.height/2)

    def __str__(self):
        string = 'rectangle of shape %.3f mm x %.3f mm'%(self.width*1e3, self.height*1e3)
        if self.center != (0, 0):
            string += 'and center (%.3f mm, %.3f mm)'%(self.center[0]*1e3, self.center[1]*1e3)
        return string

    def make_perimeter(self):
        return np.asarray([-0.5, 0.5, 0.5, -0.5, -0.5])*self.width, np.asarray([-0.5, -0.5, 0.5, 0.5, -0.5])*self.height

    def sample(self, step=None):
        if step is None:
            step = max(self.width, self.height)/16
        num_x_points = int(np.ceil(self.width/step))
        num_y_points = int(np.ceil(self.height/step))
        x = np.linspace(-0.5, 0.5, num_x_points)[:, None]*self.width + self.center[0]
        y = np.linspace(-0.5, 0.5, num_y_points)*self.height + self.center[1]
        return x, y

    def get_interval(self, origin, vector):
        ox, oy = origin
        vx, vy = vector
        dx0 = np.divide(-self.width/2 - ox, vx)
        dx1 = np.divide(self.width/2 - ox, vx)
        dy0 = np.divide(-self.height/2 - oy, vy)
        dy1 = np.divide(self.height/2 - oy, vy)
        d1 = max(min(dx0, dx1), min(dy0, dy1))
        d2 = min(max(dx0, dx1), max(dy0, dy1))
        return d1, d2

class CircleBoundary(Boundary):
    def __init__(self, diameter, center=(0, 0)):
        Boundary.__init__(self, True)
        self.diameter = diameter
        assert len(center) == 2
        self.center = center

    def __repr__(self):
        return 'CircleBoundary(diameter=%r, center=%r)'%(self.diameter, self.center)

    def is_inside(self, x, y):
        x = np.asarray(x)
        y = np.asarray(y)
        xc, yc = self.center
        return (x - xc)**2 + (y - yc)**2 <= (self.diameter/2)**2

    def __str__(self):
        string = 'circle of diameter %.3f mm'%(self.diameter*1e3)
        if self.center != (0, 0):
            string += 'and center (%.3f mm, %.3f mm)'%(self.center[0]*1e3, self.center[1]*1e3)
        return string