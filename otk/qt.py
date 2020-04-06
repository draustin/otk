import sys
from contextlib import contextmanager
from PyQt5 import QtWidgets

@contextmanager
def application():
    """Create QApplication if necessary, yield control for GUI setup, and execute QApplication if it was created.

    Using this context manager enables Qt-based scripts to run from the command line as well inside environments with
    an existing QApplication such as IPython or Spyder.

    Spyder setup: Preferences / IPython console / Graphics / Graphics backend: select Qt5.
    """
    if QtWidgets.QApplication.instance():
        app = QtWidgets.QApplication.instance()
        exec_app = False
    else:
        app = QtWidgets.QApplication(sys.argv)
        exec_app = True

    yield

    if exec_app:
        app.exec()