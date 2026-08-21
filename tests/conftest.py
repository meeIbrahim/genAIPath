import pytest

from app.pipeline.config import set_active


@pytest.fixture(autouse=True)
def _reset_active_pipeline():
    set_active(None)
    yield
    set_active(None)
