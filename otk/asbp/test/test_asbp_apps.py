import pytest
from otk.asbp.apps import SimpleLensPropagator

@pytest.mark.slow
def test_SimpleLensPropagcator(qtbot):
    app = SimpleLensPropagator()
    app.propagate()
    app.show()
    qtbot.addWidget(app)