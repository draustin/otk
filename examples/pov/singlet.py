from otk import pov, trains, ri

base_name = 'singlet'
lamb = 800e-9

with pov.open_file(base_name + '.pov') as scene:
    scene.write_includes(['glass'])

    with scene.braces('camera'):
        scene.write_vector('location', (0, 0, -6))
        scene.write_vector('look_at', (0, 0, 0))

    with scene.light_source((-5, 0, 0)):
        scene.write_vector('color', (1, 1, 1))

    num_cols = 3
    num_rows = 3
    pitch = 2
    for num, shape in enumerate(range(-4, 5)):
        thickness = 0.2 if num % 2 else 1
        singlet = trains.Singlet.from_focal_length(2, ri.fused_silica, thickness, 0.5, shape, lamb=lamb)
        x = ((num % num_cols) - (num_cols-1)/2)*pitch
        y = (int(num/num_cols) - (num_rows - 1)/2)*pitch
        shape = 'square' if num % 2 else 'circle'
        with pov.make_singlet(scene, singlet, shape):
            pov.apply_glass_material(scene, singlet.n, lamb, 'realistic')
            scene.write_vector('rotate', (0, 90, 0))
            scene.write_vector('translate', (x, y, 0))

pov.render(base_name + '.pov', base_name + '.png')