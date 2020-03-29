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

#### 2020-03-04

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