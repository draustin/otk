import pytest
from otk.asbp.apps import SimpleLensPropagator

@pytest.mark.slow
def test_SimpleLensPropagator(qtbot):
    app = SimpleLensPropagator()
    app.propagate()
    app.show()
    qtbot.addWidget(app)