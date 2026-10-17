import server


def test_state_defaults_also_read_for_old_state_files(client):
    st = server.load_state("test", server.PROFILE)
    del st["also_read"]
    server.save_state("test", st)
    assert server.load_state("test", server.PROFILE)["also_read"] == []


def test_add_and_remove_also_read(client):
    r = client.post("/api/also-read", json={"title": "Piranesi", "like": "love"})
    assert r.status_code == 200
    entries = r.get_json()["also_read"]
    assert entries[0]["title"] == "Piranesi"
    assert entries[0]["like"] == "love"
    assert entries[0]["added_at"]

    assert client.get("/api/state").get_json()["also_read"] == entries

    r = client.post("/api/also-read/remove", json={"title": "piranesi"})
    assert r.get_json()["also_read"] == []


def test_add_also_read_replaces_duplicate_title_case_insensitively(client):
    client.post("/api/also-read", json={"title": "Piranesi", "like": None})
    r = client.post("/api/also-read", json={"title": "PIRANESI", "like": "meh"})
    entries = r.get_json()["also_read"]
    assert len(entries) == 1
    assert entries[0]["like"] == "meh"


def test_add_also_read_rejects_blank_or_bad_like(client):
    assert client.post("/api/also-read", json={"title": "  ", "like": None}).status_code == 400
    assert client.post("/api/also-read", json={"title": "X", "like": "adore"}).status_code == 400


def test_fresh_picks_payload_includes_also_read(client, monkeypatch):
    client.post("/api/also-read", json={"title": "Piranesi", "like": "love"})
    client.post("/api/mark", json={"id": "hyperion", "patch": {"read": True}})

    seen = {}
    def fake_call_claude(payload, **kw):
        seen.update(payload)
        return '[{"title": "House of Suns", "author": "Alastair Reynolds", "genre": "space", "why": "x"}]'
    monkeypatch.setattr(server, "call_claude", fake_call_claude)

    assert client.post("/api/fresh-picks").status_code == 200
    assert "Piranesi" not in seen["avoidTitles"]  # loved, so listed in lovedSoFar instead
    assert seen["alsoRead"] == [{"title": "Piranesi", "reaction": "loved"}]


def test_fresh_picks_signals_loved_and_passed_from_history_and_also_read(client, monkeypatch):
    # Seed history with books replaced in earlier rounds
    st = server.load_state("test", server.PROFILE)
    st["history"] = [
        {"id": "dune", "title": "Dune", "author": "Frank Herbert", "genre": "space",
         "mark": "loved", "replaced_at": "2026-01-01T00:00:00+00:00"},
        {"id": "foundation", "title": "Foundation", "author": "Isaac Asimov", "genre": "space",
         "mark": "passed", "replaced_at": "2026-01-01T00:00:00+00:00"},
        {"id": "solaris", "title": "Solaris", "author": "Stanisław Lem", "genre": "space",
         "mark": "read", "replaced_at": "2026-01-01T00:00:00+00:00"},
    ]
    server.save_state("test", st)
    client.post("/api/also-read", json={"title": "Piranesi", "like": "love"})
    client.post("/api/also-read", json={"title": "Ringworld", "like": "meh"})
    client.post("/api/mark", json={"id": "hyperion", "patch": {"like": "love"}})

    seen = {}
    def fake_call_claude(payload, **kw):
        seen.update(payload)
        return '[{"title": "House of Suns", "author": "Alastair Reynolds", "genre": "space", "why": "x"}]'
    monkeypatch.setattr(server, "call_claude", fake_call_claude)
    assert client.post("/api/fresh-picks").status_code == 200

    assert seen["lovedSoFar"] == ["Hyperion by Dan Simmons", "Dune by Frank Herbert", "Piranesi"]
    assert seen["passedSoFar"] == ["Foundation by Isaac Asimov", "Ringworld"]
    assert "Solaris by Stanisław Lem" in seen["avoidTitles"]
    assert "Solaris by Stanisław Lem" not in seen["lovedSoFar"]
    for t in seen["lovedSoFar"] + seen["passedSoFar"]:
        assert t not in seen["avoidTitles"]
