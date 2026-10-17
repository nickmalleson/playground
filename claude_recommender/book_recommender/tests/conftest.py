import json
import pytest

import server


@pytest.fixture
def client(tmp_path, monkeypatch):
    """Flask test client with a throwaway user directory and a minimal profile."""
    monkeypatch.setattr(server, "USERS_DIR", tmp_path / "users")
    profile = {
        "display_name": "Test",
        "original_favourites": ["Ursula K. Le Guin — The Left Hand of Darkness"],
        "genres": [{"key": "space", "label": "Space opera", "sub": ""}],
        "seed_books": [
            {"id": "hyperion", "title": "Hyperion", "author": "Dan Simmons",
             "genre": "space", "why": "x"},
        ],
    }
    monkeypatch.setattr(server, "CURRENT_USER", "test")
    monkeypatch.setattr(server, "PROFILE", profile)
    (tmp_path / "users" / "test").mkdir(parents=True)
    (tmp_path / "users" / "test" / "profile.json").write_text(json.dumps(profile))
    server.app.config["TESTING"] = True
    return server.app.test_client()
