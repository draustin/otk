from otk.asbp.apps import SimpleLensPropagator

def test_SimpleLensPropagcator(qtbot):
    app = SimpleLensPropagator()
    app.propagate()
    app.show()
    qtbot.addWidget(app)