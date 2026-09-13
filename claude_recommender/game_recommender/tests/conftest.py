import json
import pytest

import server


@pytest.fixture
def client(tmp_path, monkeypatch):
    """Flask test client with a throwaway user directory and a minimal profile."""
    monkeypatch.setattr(server, "USERS_DIR", tmp_path / "users")
    profile = {
        "display_name": "Test",
        "original_favourites": ["Hollow Knight — Team Cherry"],
        "genres": [{"key": "metroidvania", "label": "Metroidvania", "sub": ""}],
        "seed_games": [
            {"id": "ori", "title": "Ori and the Blind Forest", "developer": "Moon Studios",
             "genre": "metroidvania", "why": "x"},
        ],
    }
    monkeypatch.setattr(server, "CURRENT_USER", "test")
    monkeypatch.setattr(server, "PROFILE", profile)
    (tmp_path / "users" / "test").mkdir(parents=True)
    (tmp_path / "users" / "test" / "profile.json").write_text(json.dumps(profile))
    server.app.config["TESTING"] = True
    return server.app.test_client()
