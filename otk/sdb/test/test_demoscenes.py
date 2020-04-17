from otk.sdb import sdb_qt as sdb
from otk.sdb import demoscenes

def test_demoscenes(qtbot):
    scenes = demoscenes.make_all_scenes()
    w = sdb.ScenesViewer(scenes)
    for num in range(len(scenes)):
        w.set_scene(num)
    w.show()