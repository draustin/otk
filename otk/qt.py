"""Qt-related utilities."""
import sys
from contextlib import contextmanager
from PyQt5 import QtWidgets, QtCore

@contextmanager
def application():
    """Create QApplication if necessary, yield control for GUI setup, and execute QApplication if it was created.

    Using this context manager enables Qt-based scripts to run from the command line as well inside environments with
    an existing QApplication such as IPython or Spyder.

    Spyder setup: Preferences / IPython console / Graphics / Graphics backend: select Qt5.
    """
    if not QtWidgets.QApplication.instance():
        print('QCoreApplication not created, so setting AA_EnableHighDpiScaling. Expect a warning - possible in Qt / PyQt.')
        QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)
    if QtWidgets.QApplication.instance():
        app = QtWidgets.QApplication.instance()
        exec_app = False
    else:
        app = QtWidgets.QApplication(sys.argv)
        exec_app = True

    yield

    if exec_app:
        app.exec()