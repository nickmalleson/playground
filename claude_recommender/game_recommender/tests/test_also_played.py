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


def test_fresh_picks_signals_loved_and_passed_from_history_and_also_played(client, monkeypatch):
    # Seed history with games replaced in earlier rounds
    st = server.load_state("test", server.PROFILE)
    st["history"] = [
        {"id": "firewatch", "title": "Firewatch", "developer": "Campo Santo", "genre": "metroidvania",
         "mark": "loved", "replaced_at": "2026-01-01T00:00:00+00:00"},
        {"id": "virginia", "title": "Virginia", "developer": "Variable State", "genre": "metroidvania",
         "mark": "passed", "replaced_at": "2026-01-01T00:00:00+00:00"},
        {"id": "gone-home", "title": "Gone Home", "developer": "Fullbright", "genre": "metroidvania",
         "mark": "played", "replaced_at": "2026-01-01T00:00:00+00:00"},
    ]
    server.save_state("test", st)
    client.post("/api/also-played", json={"title": "Celeste", "like": "love"})
    client.post("/api/also-played", json={"title": "Fez", "like": "meh"})
    client.post("/api/mark", json={"id": "ori", "patch": {"like": "love"}})

    seen = {}
    def fake_call_claude(payload, **kw):
        seen.update(payload)
        return '[{"title": "Axiom Verge", "developer": "Thomas Happ", "genre": "metroidvania", "why": "x"}]'
    monkeypatch.setattr(server, "call_claude", fake_call_claude)
    assert client.post("/api/fresh-picks").status_code == 200

    assert seen["lovedSoFar"] == ["Ori and the Blind Forest by Moon Studios", "Firewatch by Campo Santo", "Celeste"]
    assert seen["passedSoFar"] == ["Virginia by Variable State", "Fez"]
    assert "Gone Home by Fullbright" in seen["avoidTitles"]
    assert "Gone Home by Fullbright" not in seen["lovedSoFar"]
