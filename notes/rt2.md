# rt2 - second version of ray tracing package

This file is a roadmap and very rough introduction.

## Design specification

### Required features

High level:
* Boolean geometry operations
* Arbitrary surface profiles
* Sequential and non-sequential tracing
* In non-sequential, ray branching

### List of cases we need to handle (in the long run)

* Bonded materials
* GRIN lens
* multimode fiber
* birefringent (mono and biaxial)
* Vernet effect (Faraday)
* huge number of (implicitly defined) surfaces
* cross sections
* edge/corner rounding

### Organization

A *surface* is just a signed distance function - a mathematical object. It has an interior and an exterior.

An *element* is a surface with defined physical properties. An element has 

* a surface,
* an interior definition,
* an interface definition,
* metadata for e.g. plotting.

### Nonsequential ray tracing algorithm

Inputs are start point and an object that start point is inside, or None. If start object is None then find first intersection with union of all objects. Otherwise find first intersection with object. Intersection here means a pair of points along the ray whose SDB are both smaller in magnitude than the tolerance, but have opposite sign. Calculate reflected / and or transmitted rays starting from their respective intersection points.

## Development

### 2020-03-29

Regarding Assembly. TODO Making self.surface a simple union of the Element surfaces prevents a hierachical organization for e.g.
bounding boxes and transformations. For example with a compound lens made of different material glasses, we must
have different Elements (per current definition). But we might want these to transform together and have a common
bounding volume for e.g. participation in bounding volume optimizations such as the priority queue algorithm. Want
all this logic separate from the optics so it's available for the rendering too.

Proposed re-architecture: the geometry of the Assembly is represented by a single surface plus a mapping from Surface
to Element.  The catch is ray tracing when inside an Element. At present we only test against the Element rather than the
whole scene. Premature optimization? Perhaps could create a DisjointUnion of child Surfaces with a last_child
attribute. Its getsdb tests this child first, and if the point is inside, returns that without testing the others.
Also has lazy (bool) attribute for debugging.

The problem is position-dependent properties of an Element.

The geometrical (i.e. ignoring optics) definition of an Element is different to a Surface.

Problem with above: adjacent pieces of identical material with an interface between them. We don't model the microphysics of interfaces e.g. multilayers dielectrics. We take as given reflection and transmission vs wavelength, angle and polarization. So do need the concept of sphere trace inside an Element.

Has position-dependent properties defined in the same coordinate system as its Surface.

An Assembly has a root Surface and list of (Surface, Element) pairs.

get_element(surface, x) returns the element occupying point x.

property(surface, x) = property(surface) for primitives, property(which(surface, x), own_transform(surface, x)) for compound

E.g. refractive index, transformation, interface.

This will really complicate glsl and numba. Premature optimization? Top-level Elements only?

---

Proposed definition of Element: A region of space through which light passes without encountering an interface. Defined by a Surface. An Assembly is a set of Elements.


### 2020-03-04

Near-term feature list:

* Aspherics
  * More generally: sag profiles. Need Lipshitz
* Optical ray tracing
* cross section view
    * color
    * border
* smoothed edge union
* 2d array with smooth transition

#### Properties of a point - how to decide

If an element (i.e. has a surface and single values for each property)
For a union of disjoint elements, can define property of a point as property of the surface that contains it.


##### surface labelling scheme

Should surfaces have a 'metadata' dict attribute for keeping optical/physical/aesthetic information (option A)? Alternative is a mapping from surfaces to per-surface information for each subsystem (option B).

B keeps surfaces lightweight - no empty dictionary.



Disadvantages: the optical/physical/aesthetic code


    * label surfaces with attributes e.g. color, edge style