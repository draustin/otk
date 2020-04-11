from otk.sdb import sdb_qt as sdb
from otk.sdb import demoscenes

def test_demoscenes(qtbot):
    w = sdb.ScenesViewer(demoscenes.make_all_scenes())
    w.show()