from otk import sdb_qt as sdb
from otk.sdb import demoscenes

with sdb.application():
    w = sdb.ScenesViewer(demoscenes.make_all_scenes())
    w.show()