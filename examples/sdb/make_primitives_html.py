from otk.sdb import demoscenes, js
scene = demoscenes.make_primitives()
print(js.gen_html(scene.sdb_glsl), file=open('primitives.html', 'wt'))
