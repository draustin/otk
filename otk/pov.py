"""Tools for generating scenes in POV-Ray.

I wrote this module before the 3D geometry system based on signed distance bounds (sdb). If POV-Ray output is desired
in future it would be much better to make it a backend of sdb.
"""
import os, shutil
import subprocess
from typing import Sequence, TextIO, Tuple, Callable
import numpy as np
from contextlib import contextmanager
from . import functions, trains, ri

POVRAY_BINARY = ("povray.exe" if os.name == 'nt' else "povray")

povray_installed = shutil.which(POVRAY_BINARY) is not None

transparent_indices = {ri.vacuum, ri.air}

def render(input_file: str, output_file: str, height=None, width=None, quality=None, antialiasing=None, remove_temp=True, show_window=False,
    includedirs=None, output_alpha=False):

    cmd = [POVRAY_BINARY, input_file]
    if height is not None:
        cmd.append('+H%d'%height)
    if width is not None:
        cmd.append('+W%d'%width)
    if quality is not None:
        cmd.append('+Q%d'%quality)
    if antialiasing is not None:
        cmd.append('+A%f'%antialiasing)
    if output_alpha:
        cmd.append('Output_Alpha=on')
    if not show_window:
        cmd.append('-D')
    else:
        cmd.append('+D')
    if includedirs is not None:
        for dir in includedirs:
            cmd.append('+L%s'%dir)

    cmd.append("+O%s"%output_file)

    result = subprocess.run(cmd, stderr=subprocess.PIPE, stdin=subprocess.PIPE, stdout=subprocess.PIPE)

    if result.returncode:
        print(type(result.stderr), result.stderr)
        raise IOError("POVRay rendering failed with the following error: "+result.stderr.decode('ascii'))

class Scene:
    def __init__(self, file: TextIO, indent:int=4):
        self.file = file
        self.indent = indent
        self.level = 0
        self.loop_level = 0

    def writeline(self, line: str):
        self.file.write(' '*self.indent*self.level + line + '\n')

    @contextmanager
    def braces(self, name: str):
        self.writeline(name + ' {')
        self.level += 1
        yield
        self.level -= 1
        self.writeline('}')

    @contextmanager
    def sphere(self, center: Sequence, radius: float):
        with self.braces('sphere'):
            self.writeline(format_vector(*center))
            self.writeline(f'{radius:g}')
            yield

    def write_includes(self, names: Sequence):
        for name in names:
            self.writeline(f'#include "{name}.inc"')

    def write_comment(self, comment: str):
        self.writeline('// ' + comment)

    def write_vector(self, name, vector):
        self.writeline(name + ' ' + format_vector(*vector))

    @contextmanager
    def light_source(self, location: Sequence):
        with self.braces('light_source'):
            self.writeline(format_vector(*location))
            yield

    @contextmanager
    def cylinder(self, base: Sequence, cap:Sequence, radius: float):
        with self.braces('cylinder'):
            self.writeline(format_vector(*base))
            self.writeline(format_vector(*cap))
            self.writeline(f'{radius:g}')
            yield

    @contextmanager
    def box(self, corner1: Sequence, corner2: Sequence, *args):
        with self.braces('box'):
            self.writeline(format_vector(*corner1))
            self.writeline(format_vector(*corner2))
            yield

    @contextmanager
    def for_loop(self, identifier: str, start, stop, step=1):
        self.writeline(f'#for ({identifier}, {start:g}, {stop:g}, {step:g})')
        self.level += 1
        yield
        self.level -= 1
        self.writeline('#end')

    @contextmanager
    def make_2d_array(self, pitch, num_side: int):
        i0 = (num_side - 1)/2
        ix = f'ix{self.loop_level:d}'
        iy = f'iy{self.loop_level:d}'
        self.loop_level += 1
        with self.braces('union'):
            with self.for_loop(ix, 0, num_side - 1, 1):
                with self.for_loop(iy, 0, num_side - 1, 1):
                    with self.braces('object'):
                        yield
                        self.writeline(f'translate <({ix}-{i0:g})*{pitch:g}, ({iy}-{i0:g})*{pitch:g}, 0>')
        self.loop_level -= 1

    def colored_texture(self, r, g, b):
        with self.braces('texture'):
            with self.braces('pigment'):
                self.write_vector('color', (r, g, b))

    def verbatim(self, text: str):
        self.file.write(text)

    def polygon(self, points):
        with self.braces('polygon'):
            self.writeline(f'{len(points):d}, ' + ', '.join(f'<{x:g}, {y:g}>' for x, y in points))


@contextmanager
def open_file(filename: str, indent: int=4):
    with open(filename, 'wt') as file:
        scene = Scene(file, indent)
        yield scene

def format_vector(*args):
    return '<' + ', '.join(f'{x:g}' for x in args) + '>'


@contextmanager
def spherical_lens(scene, roc1: float, roc2: float, thickness: float, radius: float, shape: str = 'circle'):
    sag1 = functions.calc_sphere_sag(roc1, radius)
    sag2 = functions.calc_sphere_sag(roc2, radius)

    scene.write_comment('square_spherical_lens')

    z1 = min(sag1, 0)
    z2 = thickness + max(sag2, 0)
    def make_body():
        if shape == 'square':
            hsl = radius/2**0.5
            with scene.box([-hsl, -hsl, z1], [hsl, hsl, z2]):
                pass
        elif shape == 'circle':
            with scene.cylinder([0, 0, z1], [0, 0, z2], radius):
                pass

    def make_left():
        if np.isfinite(roc1):
            with scene.braces('intersection' if roc1 > 0 else 'difference'):
                # Make body and cut first surface out of it.
                make_body()
                with scene.braces('union'):
                    with scene.sphere([0, 0, roc1], abs(roc1)):
                        pass
                    if roc1 > 0:
                        with scene.cylinder((0, 0, sag1), (0, 0, thickness + max(sag2, 0)), radius):
                            pass
        else:
            make_body()

    with scene.braces('object'):
        if np.isfinite(roc2):
            with scene.braces('intersection' if roc2 < 0 else 'difference'):
                # Make body with first surface cut out, and then cut out right surface.
                make_left()
                with scene.braces('union'):
                    with scene.sphere([0, 0, thickness + roc2], abs(roc2)):
                        pass
                    if roc2 < 0:
                        with scene.cylinder((0, 0, min(sag1, 0)), (0, 0, thickness + sag2), radius):
                            pass
        else:
            make_left()

        # I am not sure if this speeds things up. See  https://www.povray.org/documentation/view/3.6.1/323/ and
        # http://www.povray.org/documentation/view/3.6.1/184/. Large lens arrays implemented as unions slow POV-Ray down,
        # which seems inconsistent with then manual's claim that it handles unions efficiently. I did not test carefully
        # whether this line makes a difference, but it doesn't seem to cause any harm.
        with scene.braces('bounded_by'):
            make_body()

        yield

@contextmanager
def make_singlet(scene, singlet: trains.Singlet, shape: str):
    with spherical_lens(scene, singlet.surfaces[0].roc, singlet.surfaces[1].roc, singlet.thickness, singlet.radius, shape):
            yield

@contextmanager
def make_singlet_sequence(scene, sequence: trains.SingletSequence, shape: str, apply_material: Callable[[Scene, int, ri.Index], None]=None):
    z = sequence.spaces[0]

    with scene.braces('union'):
        for num, (singlet, space) in enumerate(zip(sequence.singlets, sequence.spaces[1:])):
            with make_singlet(scene, singlet, shape):
                if apply_material is not None:
                    apply_material(scene, num, singlet.n)
                scene.write_vector('translate', [0, 0, z])
            z += singlet.thickness + space

        yield

@contextmanager
def make_train(scene, train: trains.Train, shape: str, radii='equal', apply_material: Callable=None):
    """From first to last interface."""
    z = train.spaces[0]

    with scene.braces('union'):
        for num, (i1, i2, thickness) in enumerate(zip(train.interfaces[:-1], train.interfaces[1:], train.spaces[1:])):
            n = i1.n2
            if n not in transparent_indices:
                if radii == 'equal':
                    radius = i1.radius
                    assert i2.radius == radius
                elif radii == 'max':
                    radius = max((i1.radius, i2.radius))
                else:
                    radius = radii

                with spherical_lens(scene, i1.roc, i2.roc, thickness, radius, shape):
                    if apply_material is not None:
                        apply_material(scene, num, n)
                    scene.write_vector('translate', (0, 0, z))

            z += thickness

        yield

def apply_glass_realistic(scene: Scene, n:float, r:float, g:float, b:float):
    """Apply somewhat realistic glass properties.

    Color values should be close to 1 for a realistic 'tinge'.

    I couldn't find a good reference on POV-Ray's model, but through experimentation found that the fourth
    component of color rgbf affects specular highlights and diffuse. Note that specular highlights apply to
    light sources - this is different from (specular) reflection which applies to light scattered from other objects.
    After a bit of experimentation I converged on this as a decent semi-realistic model for optical lens glass.
    """
    with scene.braces('texture'):
        with scene.braces('pigment'):
            scene.write_vector('color rgbf', (r, g, b, 0.99))

        with scene.braces('finish'):
            scene.writeline('specular 0.8')
            scene.writeline('roughness 0.002')
            scene.writeline('ambient 0')
            scene.writeline('diffuse 0.2')
            with scene.braces('reflection'):
                scene.writeline('1')
                scene.writeline('fresnel on')

            scene.writeline('conserve_energy')

    with scene.braces('interior'):
        scene.writeline(f'ior {n:g}')


def apply_glass_material(scene: Scene, n:ri.Index, lamb: float, style: str):
    assert style in ('realistic', 'CAD')
    if style == 'realistic':
        color = {'fused_silica': (0.95, 0.95, 1), 'K-VC89':(1, 0.95, 0.95), 'N-BK7':(0.95, 0.95, 1)}[n.name]
        apply_glass_realistic(scene, n(lamb), *color)
    else:
        # CAD style is opaque & designed for a white backbround.
        color = {'fused_silica': (0, 0, 1), 'K-VC89': (1, 0, 0), 'N-BK7': (0, 0.5, 1)}[n.name]
        # Doesn's seem to work. CAD style means 'shadowless' light sources, which does work.
        #scene.writeline('no_shadow')
        with scene.braces('texture'):
            with scene.braces('pigment'):
                scene.write_vector('color', color)

            with scene.braces('finish'):
                scene.writeline('ambient 0.1')
                scene.writeline('diffuse 0.8')
                scene.writeline('specular 0')
                #scene.writeline('diffuse 0.2')