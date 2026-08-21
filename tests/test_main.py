import uuid

from app.config import Settings
from app.main import create_app


def _isolated_settings(tmp_path) -> Settings:
    return Settings(
        qdrant_path=str(tmp_path / "qdrant"),
        qdrant_collection=f"test-{uuid.uuid4()}",
        vector_size=2,
    )


def test_create_app_registers_expected_routes(tmp_path):
    app = create_app(_isolated_settings(tmp_path))
    paths = set(app.openapi()["paths"].keys())
    assert "/query" in paths
    assert "/pipeline/load" in paths
    assert "/pipeline/status" in paths
