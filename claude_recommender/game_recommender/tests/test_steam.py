import json
import time

import pytest

import steam


# ── matching ──

@pytest.mark.parametrize("title,candidates,expected", [
    ("Helldivers 2", ["HELLDIVERS™ 2", "HELLDIVERS™ 2 - TR-117 Alpha Commander Armor Set"], 0),
    ("Kentucky Route Zero", ["Kentucky Route Zero: PC Edition"], 0),
    ("Hollow Knight", ["Hollow Knight: Silksong", "Hollow Knight"], 1),
    ("Hollow Knight", ["Hollow Knight: Silksong"], None),        # a different game, not an edition
    ("Ori and the Blind Forest", ["Ori and the Blind Forest: Definitive Edition"], 0),
    ("NORCO", ["NORCO"], 0),
    ("Serious Sam", ["Serious Sam 4", "Serious Sam: The First Encounter"], None),
    ("Neon White", ["Neon White Soundtrack"], None),
    ("Deep Rock Galactic", ["Deep Rock Galactic", "Deep Rock Galactic: Rogue Core"], 0),
    ("Made-up Game 9000", ["Something Else Entirely"], None),
])
def test_pick_match(title, candidates, expected):
    items = [{"id": i, "name": n} for i, n in enumerate(candidates)]
    got = steam.pick_match(title, items)
    assert (got["id"] if got else None) == expected


# ── lookup + cache ──

def fake_fetch_factory(search, reviews):
    def fake_fetch_json(url):
        if "storesearch" in url:
            return search
        if "appreviews" in url:
            return reviews
        raise AssertionError(url)
    return fake_fetch_json


def test_lookup_returns_rating_record(monkeypatch):
    monkeypatch.setattr(steam, "_fetch_json", fake_fetch_factory(
        {"items": [{"id": 553850, "name": "HELLDIVERS™ 2"}]},
        {"query_summary": {"total_positive": 877948, "total_negative": 284070,
                           "total_reviews": 1162018, "review_score_desc": "Mostly Positive"}},
    ))
    rec = steam.lookup("Helldivers 2")
    assert rec == {
        "appid": 553850, "name": "HELLDIVERS™ 2",
        "positive": 877948, "total": 1162018, "pct": 76,
        "score_desc": "Mostly Positive",
        "url": "https://store.steampowered.com/app/553850",
    }


def test_lookup_returns_none_when_no_match(monkeypatch):
    monkeypatch.setattr(steam, "_fetch_json", fake_fetch_factory({"items": []}, {}))
    assert steam.lookup("Made-up Game 9000") is None


def test_lookup_returns_none_when_no_reviews(monkeypatch):
    monkeypatch.setattr(steam, "_fetch_json", fake_fetch_factory(
        {"items": [{"id": 1, "name": "Tiny Game"}]},
        {"query_summary": {"total_positive": 0, "total_negative": 0, "total_reviews": 0}},
    ))
    assert steam.lookup("Tiny Game") is None


def test_lookup_swallows_network_errors(monkeypatch):
    def boom(url):
        raise OSError("no network")
    monkeypatch.setattr(steam, "_fetch_json", boom)
    assert steam.lookup("Helldivers 2") is None


def test_cached_lookup_hits_network_once_and_persists(tmp_path, monkeypatch):
    calls = []
    def fake(url):
        calls.append(url)
        if "storesearch" in url:
            return {"items": [{"id": 1, "name": "Neon White"}]}
        return {"query_summary": {"total_positive": 9, "total_negative": 1, "total_reviews": 10,
                                  "review_score_desc": "Positive"}}
    monkeypatch.setattr(steam, "_fetch_json", fake)
    cache = steam.SteamCache(tmp_path / "steam_cache.json")

    first = cache.get("Neon White")
    second = cache.get("neon white")   # key is case-insensitive
    assert first["pct"] == 90 and second == first
    assert len(calls) == 2  # one search + one reviews call, then cached

    # Persisted: a fresh instance reads it back without the network
    calls.clear()
    assert steam.SteamCache(tmp_path / "steam_cache.json").get("Neon White") == first
    assert calls == []


def test_cached_miss_is_retried_after_ttl(tmp_path, monkeypatch):
    calls = []
    def fake(url):
        calls.append(url)
        return {"items": []}
    monkeypatch.setattr(steam, "_fetch_json", fake)
    cache = steam.SteamCache(tmp_path / "steam_cache.json")

    assert cache.get("Nowhere Game") is None
    assert cache.get("Nowhere Game") is None
    assert len(calls) == 1  # miss is cached too

    # Age the miss past the TTL
    data = json.loads((tmp_path / "steam_cache.json").read_text())
    data["nowhere game"]["checked_at"] = time.time() - steam.MISS_TTL_S - 1
    (tmp_path / "steam_cache.json").write_text(json.dumps(data))
    assert steam.SteamCache(tmp_path / "steam_cache.json").get("Nowhere Game") is None
    assert len(calls) == 2


# ── endpoint ──

def test_api_steam_endpoint(client, monkeypatch):
    import server
    monkeypatch.setattr(server.STEAM, "get", lambda title: {"pct": 92, "total": 41000} if title == "Neon White" else None)
    assert client.get("/api/steam?title=Neon%20White").get_json() == {"ok": True, "rating": {"pct": 92, "total": 41000}}
    assert client.get("/api/steam?title=Nope").get_json() == {"ok": True, "rating": None}
    assert client.get("/api/steam").status_code == 400
