from otk import pov, trains, ri

base_name = 'singlet_sequence'
lamb = 800e-9
half = trains.SingletSequence.design_symmetric_singlet_transform(ri.fused_silica, 5, 0.5, 0.1, 1, lamb=lamb)
sequence = half + half.reverse()
print(sequence)

with pov.open_file(base_name + '.pov') as scene:
    scene.write_includes(['glass'])

    with scene.braces('camera'):
        scene.write_vector('location', (0, 0, -8))
        scene.write_vector('look_at', (0, 0, 0))

    with scene.light_source((-5, 0, 0)):
        scene.write_vector('color', (1, 1, 1))

    for shape, y in (('square', -1), ('circle', 1)):
        with pov.make_singlet_sequence(scene, sequence, shape, lambda scene, num, n: pov.apply_glass_material(scene, n, lamb, 'realistic')):
            scene.write_vector('translate', (0, y, -sequence.center_length/2))
            scene.write_vector('rotate', (0, 45, 0))

pov.render(base_name + '.pov', base_name + '.png')