from app.sessions import SessionStore


class FakeClock:
    def __init__(self, now: float = 1_000_000.0):
        self.now = now

    def __call__(self) -> float:
        return self.now


TTL = 3600.0


def make_store(tmp_path, clock=None) -> SessionStore:
    return SessionStore(tmp_path / "sessions.db", ttl_seconds=TTL, clock=clock or FakeClock())


# --- SessionStore ---------------------------------------------------------


def test_create_and_get(tmp_path):
    store = make_store(tmp_path)
    s = store.create()
    assert store.get(s.id) == s


def test_ids_are_unique(tmp_path):
    store = make_store(tmp_path)
    assert len({store.create().id for _ in range(50)}) == 50


def test_unknown_session_returns_none(tmp_path):
    assert make_store(tmp_path).get("nope") is None


def test_session_expires_after_ttl_of_inactivity(tmp_path):
    clock = FakeClock()
    store = make_store(tmp_path, clock)
    s = store.create()
    clock.now += TTL + 1
    assert store.get(s.id) is None


def test_activity_extends_lifetime(tmp_path):
    clock = FakeClock()
    store = make_store(tmp_path, clock)
    s = store.create()
    for _ in range(3):
        clock.now += TTL - 10
        assert store.get(s.id) is not None  # each lookup refreshes last_active_at


def test_purge_removes_only_expired(tmp_path):
    clock = FakeClock()
    store = make_store(tmp_path, clock)
    old = store.create()
    clock.now += TTL + 1
    fresh = store.create()
    assert store.purge_expired() == [old.id]
    assert store.get(fresh.id) is not None
    assert store.purge_expired() == []


def test_delete(tmp_path):
    store = make_store(tmp_path)
    s = store.create()
    assert store.delete(s.id) is True
    assert store.get(s.id) is None
    assert store.delete(s.id) is False


def test_sessions_survive_restart(tmp_path):
    clock = FakeClock()
    s = make_store(tmp_path, clock).create()
    reopened = make_store(tmp_path, clock)
    assert reopened.get(s.id) is not None


# --- API --------------------------------------------------------------------


def test_create_session_endpoint(make_client):
    client = make_client()
    resp = client.post("/api/sessions")
    assert resp.status_code == 201
    body = resp.json()
    assert set(body) == {"session_id", "created_at", "last_active_at", "expires_at"}
    assert client.get(f"/api/sessions/{body['session_id']}").status_code == 200


def test_get_unknown_session_is_404(make_client):
    resp = make_client().get("/api/sessions/does-not-exist")
    assert resp.status_code == 404
    assert resp.json()["detail"] == "Session not found or expired"


def test_delete_session_endpoint(make_client):
    client = make_client()
    sid = client.post("/api/sessions").json()["session_id"]
    assert client.delete(f"/api/sessions/{sid}").status_code == 204
    assert client.get(f"/api/sessions/{sid}").status_code == 404
    assert client.delete(f"/api/sessions/{sid}").status_code == 404


def test_expired_sessions_purged_on_startup(make_client, make_settings):
    settings = make_settings()
    clock = FakeClock()
    store = SessionStore(settings.data_dir / "sessions.db", TTL, clock)
    old = store.create()
    store.close()

    client = make_client()  # same data_dir; real clock is far past FakeClock's time
    assert client.get(f"/api/sessions/{old.id}").status_code == 404
    assert client.app.state.purge_expired_sessions() == []  # already purged at startup
