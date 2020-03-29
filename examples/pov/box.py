from otk import pov

base_name = 'box'

with pov.open_file(base_name + '.pov') as scene:
    with scene.braces('camera'):
        scene.write_vector('location', (0, 0, -4))
        scene.write_vector('look_at', (0, 0, 0))

    with scene.light_source((-5, 0, 0)):
        scene.write_vector('color', (1, 1, 1))

    with scene.box((-1, -1, -1), (1, 1, 1), 1):
        with scene.braces('texture'):
            with scene.braces('pigment'):
                scene.write_vector('color', (1, 0, 0))
        scene.write_vector('rotate', (45, 45, 0))

pov.render(base_name + '.pov', base_name + '.png')