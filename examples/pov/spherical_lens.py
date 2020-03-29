from otk import pov

base_name = 'spherical_lens'

with pov.open_file(base_name + '.pov') as scene:
    scene.write_includes(['glass'])

    with scene.braces('camera'):
        scene.write_vector('location', (0, 0, -4))
        scene.write_vector('look_at', (0, 0, 0))

    with scene.light_source((-5, 0, 0)):
        scene.write_vector('color', (1, 1, 1))

    for shape, x in (('circle', -1), ('square', 1)):
        with pov.spherical_lens(scene, 10, -10, 0.1, 1, shape):
            pov.apply_glass_realistic(scene, 1.5, 0.8, 0.8, 1)
            scene.write_vector('rotate', (0, 45, 0))
            scene.write_vector('translate', (x, 0, 0))

pov.render(base_name + '.pov', base_name + '.png')