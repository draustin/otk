"""Drawing functions using the gizeh library as a backend."""
import numpy as np
import gizeh
from . import trains, functions

def make_interface(face:trains.Interface, num_points=32, **kwargs):
    xys = face.get_points(num_points)
    element = gizeh.polyline(xys, **kwargs)
    return element

def label_interface(face:trains.Interface, radius_factor=1, font_family='sans', font_size=None, **kwargs):
    """Make Gizeh elements labelling interface.

    Consist of text saying ROC, and a line from the edge of the interface to the text.

    Args:
        radius_factor (scalar): Factor by which interface radius is multiplied to get text x position.
        font_family: Passed to gizeh.text.
        font_size:  Passed to gizeh.text.
        **kwargs:  Passed to gizeh.text.

    Returns:
        tuple of gizeh.Element: In drawing order.
    """
    if font_size is None:
        font_size = face.radius/3
    x = radius_factor*face.radius
    y = functions.calc_sphere_sag(face.roc, face.radius)
    string = 'ROC %.3f mm'%(face.roc*1e3)
    text_element = gizeh.text(string, font_family, font_size, **kwargs).translate((x, y))
    line_element = gizeh.polyline(((face.radius, y), (x, y)), stroke=(0.8, 0.8, 0.8), stroke_width=font_size/16)
    return line_element, text_element

def make_train_interfaces(train:trains.Train, **kwargs):
    """
    Propagation is along positive y axis, starting from origin.

    Args:
        size (scalar): Full transverse dimension of interfaces.

    Returns:
        gizeh.Element
    """
    y = train.spaces[0]
    elements = []
    for interface, space in zip(train.interfaces, train.spaces[1:]):
        elements.append(interface.make_gizeh_interface(**kwargs).translate((0, y)))
        y += space
    element = gizeh.Group(elements)
    return element

def make_train_spaces(train:trains.Train):
    radius = train.interfaces[0].radius  # if len(self.interfaces) > 0 else self.spaces[0]
    y_next = 0
    xys_last = np.asarray([[-radius, 0], [radius, 0]])
    n = train.interfaces[0].n1
    elements = []
    for space, next_interface in zip(train.spaces, train.interfaces + (None,)):
        y_next += space

        # Generate next interface points.
        if next_interface is None:
            xys_next = np.asarray([[-radius, 0], [radius, 0]])
        else:
            xys_next = next_interface.get_points()

        xys_next += [0, y_next]

        xys = np.r_[xys_last, xys_next[::-1, :]]
        elements.append(gizeh.polyline(xys, fill=n.section_color))

        xys_last = xys_next
        if next_interface is not None:
            n = next_interface.n2
    return gizeh.Group(elements)

def label_train_interfaces(train:trains.Train, radius_factors=1, font_family='sans', font_size=None, **kwargs):
    radius_factors = np.broadcast_to(radius_factors, (len(train.interfaces),))
    elementss = []

    y = train.spaces[0]
    for interface, space, radius_factor in zip(train.interfaces, train.spaces[1:], radius_factors):
        elements = interface.make_gizeh_label(radius_factor, font_family, font_size, **kwargs)
        elementss.append([e.translate((0, y)) for e in elements])
        y += space

    groups = [gizeh.Group(elements) for elements in zip(*elementss)]
    return groups

def label_train_spaces(train: trains.Train, radius_factors=1, font_family='sans', font_size=None, **kwargs):
    """Make Gizeh element containing text showing the material name and thickness of the spaces.

    Args:
        radius_factors (scalar or sequence): Factor by which the average interface radii is multiplied to place the text. If
            a scalar is given it is broadcast.
        font_family (str): Font family.
        font_size (scalar): Font size.
        **kwargs: Passed to gizeh.text.

    Returns:
        gizeh.Element
    """
    radius_factors = np.broadcast_to(radius_factors, (len(train.spaces),))
    if font_size is None:
        font_size = max(i.radius for i in train.interfaces)/3
    radius_last = train.interfaces[0].radius
    y_last = 0
    h_last = 0
    n = train.interfaces[0].n1
    text_elements = []
    line_elements = []
    for space, next_interface, radius_factor in zip(train.spaces, train.interfaces + (None,), radius_factors):
        if next_interface is None:
            radius_next = radius_last
            h_next = 0
        else:
            radius_next = next_interface.radius
            h_next = math.calc_sphere_sag(next_interface.roc, radius_next)
        y_next = y_last + space
        if space != 0:
            string = '%.3f mm %s'%(space*1e3, n.name)
            radius = (radius_last + radius_next)/2
            x = radius_factor*radius
            y = (y_next + h_next + y_last + h_last)/2
            text_elements.append(gizeh.text(string, font_family, font_size, **kwargs).translate((x, y)))
            line_elements.append(
                gizeh.polyline(((radius, y), (x, y)), stroke=(0.8, 0.8, 0.8), stroke_width=font_size/16))
        y_last = y_next
        h_last = h_next
        if next_interface is not None:
            n = next_interface.n2
            radius_last = next_interface.radius
    text_group = gizeh.Group(text_elements)
    line_group = gizeh.Group(line_elements)
    return text_group, line_group