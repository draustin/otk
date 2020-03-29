from otk import pov

base_name = 'spherical_lens_array'

with pov.open_file(base_name + '.pov') as scene:
    scene.write_includes(['glass'])

    with scene.braces('camera'):
        scene.write_vector('location', (0, 0, -4))
        scene.write_vector('look_at', (0, 0, 0))

    with scene.light_source((-5, 0, 0)):
        scene.write_vector('color', (1, 1, 1))

    pitch = 1
    size = 3

    with scene.braces('object'):
        with scene.make_2d_array(pitch, size):
            with pov.spherical_lens(scene, 3, -3, 0.2, 1/2**0.5, 'square'):
                pass
        pov.apply_glass_realistic(scene, 1.5, 0.8, 0.8, 1)
        scene.write_vector('rotate', (0, 45, 0))

pov.render(base_name + '.pov', base_name + '.png')