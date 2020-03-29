import numpy as np
from otk import pov, trains, ri

base_name = 'train'
lamb = 800e-9

grid_text = """\
#macro Raster(RScale, RLine)
pigment{
   gradient x scale RScale
   color_map{
     [0.000   color rgb<1,1,1>*0.5]
     [0+RLine color rgb<1,1,1>*0.5]
     [0+RLine color rgbt<1,1,1,1>]
     [1-RLine color rgbt<1,1,1,1>]
     [1-RLine color rgb<1,1,1>*0.5]
     [1.000   color rgb<1,1,1>*0.5]
            }
       } // end of pigment
#end// of "Raster(RScale, RLine)"

// -------------------<<<< Grid macro
#macro Grid(RasterScale,
            RasterHalfLine,
            Background_pigment)
plane{<0, 0, 1>, 0
       //layered textures!!!!
      texture{ Background_pigment
             } //  base color
      texture{ Raster(RasterScale,
                      RasterHalfLine)
             } // 2nd layer
      texture{ Raster(RasterScale,
                      RasterHalfLine)
               rotate<0,0,90>
             } // 3rd layer
     } // end of plane
#end
"""


with pov.open_file(base_name + '.pov') as scene:
    scene.write_includes(['glass'])

    with scene.braces('camera'):
        scene.write_vector('location', (0, 0, -8))
        scene.write_vector('look_at', (0, 0, 0))

    with scene.light_source((0, 2, -8)):
        scene.write_vector('color', (1, 1, 1))

    scene.verbatim(grid_text)

    num_cols = 3
    num_rows = 3
    pitch = 3
    for num, shape in enumerate(range(-4, 5)):
        train = trains.Train.design_singlet(ri.fused_silica, 5, shape, 0.2, 1, lamb) + trains.Train([], [1]) + trains.Train.design_singlet(ri.KVC89, -4, shape, 0.2, 0.8, lamb)

        print(train)
        x = ((num % num_cols) - (num_cols-1)/2)*pitch
        y = (int(num/num_cols) - (num_rows - 1)/2)*pitch
        shape = 'square' if num % 2 else 'circle'
        with pov.make_train(scene, train, shape, 'equal', lambda scene, num, n: pov.apply_glass_material(scene, n, lamb, 'realistic')):
            scene.write_vector('rotate', (0, 90, 0))
            scene.write_vector('translate', (x - train.length/2, y, 0))

    with scene.braces('object'):
        scene.writeline('Grid(0.5, 0.05, pigment{color rgb<1,1,1>*1.1})')
        scene.write_vector('translate', (0, 0, 3))

pov.render(base_name + '.pov', base_name + '.png')