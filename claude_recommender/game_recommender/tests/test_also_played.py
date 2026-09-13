import server


def test_state_defaults_also_played_for_old_state_files(client):
    st = server.load_state("test", server.PROFILE)
    del st["also_played"]
    server.save_state("test", st)
    assert server.load_state("test", server.PROFILE)["also_played"] == []


def test_add_and_remove_also_played(client):
    r = client.post("/api/also-played", json={"title": "Celeste", "like": "love"})
    assert r.status_code == 200
    entries = r.get_json()["also_played"]
    assert entries[0]["title"] == "Celeste"
    assert entries[0]["like"] == "love"
    assert entries[0]["added_at"]

    assert client.get("/api/state").get_json()["also_played"] == entries

    r = client.post("/api/also-played/remove", json={"title": "celeste"})
    assert r.get_json()["also_played"] == []


def test_add_also_played_replaces_duplicate_title_case_insensitively(client):
    client.post("/api/also-played", json={"title": "Celeste", "like": None})
    r = client.post("/api/also-played", json={"title": "CELESTE", "like": "meh"})
    entries = r.get_json()["also_played"]
    assert len(entries) == 1
    assert entries[0]["like"] == "meh"


def test_add_also_played_rejects_blank_or_bad_like(client):
    assert client.post("/api/also-played", json={"title": "  ", "like": None}).status_code == 400
    assert client.post("/api/also-played", json={"title": "X", "like": "adore"}).status_code == 400


def test_fresh_picks_payload_includes_also_played(client, monkeypatch):
    client.post("/api/also-played", json={"title": "Celeste", "like": "love"})
    client.post("/api/mark", json={"id": "ori", "patch": {"played": True}})

    seen = {}
    def fake_call_claude(payload, **kw):
        seen.update(payload)
        return '[{"title": "Axiom Verge", "developer": "Thomas Happ", "genre": "metroidvania", "why": "x"}]'
    monkeypatch.setattr(server, "call_claude", fake_call_claude)

    assert client.post("/api/fresh-picks").status_code == 200
    assert "Celeste" in seen["avoidTitles"]
    assert seen["alsoPlayed"] == [{"title": "Celeste", "reaction": "loved"}]
