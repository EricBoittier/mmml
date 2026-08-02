"""Regression test for GH #127 ("mmml gui not compiling"): the root cause
was a bare 404 on ``GET /`` when the frontend hasn't been built (static_dir
missing/empty), which reads as "the GUI is broken" rather than "run npm
install && npm run build". The fix makes that state an explicit, actionable
503 instead of falling through to FastAPI's default 404."""

from __future__ import annotations

from fastapi.testclient import TestClient

from mmml.gui.api.main import create_app


def test_root_explains_missing_frontend_instead_of_404(tmp_path):
    app = create_app(data_dir=str(tmp_path), static_dir=None)
    client = TestClient(app)

    response = client.get("/")

    assert response.status_code == 503
    assert "npm install" in response.text
    assert "npm run build" in response.text


def test_root_explains_missing_frontend_when_static_dir_does_not_exist(tmp_path):
    app = create_app(data_dir=str(tmp_path), static_dir=str(tmp_path / "no-such-dist"))
    client = TestClient(app)

    response = client.get("/")

    assert response.status_code == 503


def test_root_serves_built_frontend_when_static_dir_present(tmp_path):
    static_dir = tmp_path / "dist"
    static_dir.mkdir()
    (static_dir / "index.html").write_text("<html><body>viewer</body></html>")

    app = create_app(data_dir=str(tmp_path), static_dir=str(static_dir))
    client = TestClient(app)

    response = client.get("/")

    assert response.status_code == 200
    assert "viewer" in response.text


def test_api_routes_available_even_without_built_frontend(tmp_path):
    app = create_app(data_dir=str(tmp_path), static_dir=None)
    client = TestClient(app)

    response = client.get("/api/files")

    assert response.status_code == 200
