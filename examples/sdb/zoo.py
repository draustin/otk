from PyQt5 import QtWidgets
from otk.sdb import demoscenes
from otk.sdb.qt import *

app = QtWidgets.QApplication([])
w = ScenesViewer(demoscenes.make_all_scenes())
w.show()
app.exec()