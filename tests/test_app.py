from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def _run():
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    return at


def test_app_predicts_blank_form():
    at = _run()
    at.button[0].click().run()
    assert not at.exception
    assert any(m.label == "Bankruptcy probability" for m in at.metric)
    assert at.success and not at.error      # unknown values -> typical company -> low risk


def test_app_healthy_example_is_low_risk():
    at = _run()
    at.selectbox[0].set_value("Healthy company").run()
    at.button[0].click().run()
    assert not at.exception
    assert at.success


def test_app_example_and_every_horizon():
    at = _run()
    for h in [1, 2, 3, 4, 5]:
        at.radio[0].set_value(h).run()
        at.selectbox[0].set_value("Distressed company").run()
        at.button[0].click().run()
        assert not at.exception
        assert at.error, f"distressed example not flagged for {h}y"
