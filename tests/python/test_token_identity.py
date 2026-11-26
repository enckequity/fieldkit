"""Bearer-token identity and the public_url origin guard (TeamShift fork).

A middleware stands in for the engine's token middleware (hermes_cli/dashboard_auth/token_auth.py): for the
``X-Test-Token`` header it attaches ``request.state.token_principal`` (a TokenPrincipal-shaped object) and
``token_authenticated``. ``X-Test-User`` still attaches a dashboard session, as in test_api.py. Every other header is
what a phone could send, and none of them may set identity.
"""
import importlib.util
import json
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[2]
from fieldkit_core import db  # noqa: E402
from seed import seed  # noqa: E402

PREFIX = "/api/plugins/fieldkit"


@dataclass(frozen=True)
class TokenPrincipal:  # same fields as hermes_cli.dashboard_auth.base.TokenPrincipal
    principal: str
    provider: str
    scopes: tuple = ()


def _load_router():
    spec = importlib.util.spec_from_file_location("hermes_dashboard_plugin_fieldkit_token_test", ROOT / "dashboard" / "plugin_api.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _app(auth_required):
    mod = _load_router()
    app = FastAPI()
    app.state.auth_required = auth_required

    @app.middleware("http")
    async def fake_engine(request: Request, call_next):
        tok = request.headers.get("x-test-token")
        if tok:
            spec = json.loads(tok)
            request.state.token_principal = TokenPrincipal(spec["p"], spec.get("provider", "teamshift-field"),
                                                           tuple(spec.get("scopes", ["fieldkit"])))
            request.state.token_authenticated = spec.get("authenticated", True)
            if spec.get("peer"):
                request.state.peer_principal = request.state.token_principal
        user = request.headers.get("x-test-user")
        if user:
            request.state.session = SimpleNamespace(user_id=user, display_name=user.title(), provider="test")
        return await call_next(request)

    app.include_router(mod.router, prefix=PREFIX)
    return TestClient(app)


@pytest.fixture()
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("FIELDKIT_DATA_DIR", str(tmp_path))
    monkeypatch.delenv("FIELDKIT_PUBLIC_URL", raising=False)
    s = db.Store(path=str(tmp_path / "test.db"))
    db.set_store(s)
    yield s
    db.set_store(None)


def T(member, **kw):
    return {"X-Test-Token": json.dumps({"p": member, **kw})}


FORGED = {"X-Forwarded-User": "mallory", "X-Remote-User": "mallory", "X-Auth-Request-User": "mallory",
          "X-Fieldkit-User": "mallory", "X-TeamShift-Claim": "mallory", "X-User-Id": "mallory",
          "Authorization": "Bearer mallory"}


@pytest.mark.parametrize("auth_required", [True, False])
def test_token_principal_is_the_member(store, auth_required):
    c = _app(auth_required)
    me = c.get(f"{PREFIX}/me", headers=T("member:ana"))
    assert me.status_code == 200
    assert me.json()["user_id"] == "member:ana" and me.json()["provider"] == "teamshift-field"


def test_token_principal_wins_over_session(store):
    c = _app(True)
    assert c.get(f"{PREFIX}/me", headers={**T("member:ana"), "X-Test-User": "owner-session"}).json()["user_id"] == "member:ana"


@pytest.mark.parametrize("bad", [
    {"scopes": []},                       # no scope: a token minted for another route
    {"scopes": ["actions"]},              # another route's scope
    {"authenticated": False},             # principal without the engine's authenticated flag
    {"peer": True},                       # an agent's peer pass, not a member
])
@pytest.mark.parametrize("auth_required", [True, False])
def test_unusable_token_is_refused_never_local_or_session(store, bad, auth_required):
    c = _app(auth_required)
    for extra in ({}, {"X-Test-User": "owner-session"}):
        r = c.get(f"{PREFIX}/me", headers={**T("member:ana", **bad), **extra})
        assert r.status_code == 403, (bad, extra, r.text)


@pytest.mark.parametrize("pid", ["local", "owner", "", " member:ana", "member:ana "])
@pytest.mark.parametrize("auth_required", [True, False])
def test_reserved_or_malformed_principal_is_refused(store, pid, auth_required):
    r = _app(auth_required).get(f"{PREFIX}/me", headers=T(pid))
    assert r.status_code == 403


def test_client_headers_never_set_identity(store):
    loopback = _app(False)
    # No token: the loopback dashboard's single user, whatever the phone claims.
    assert loopback.get(f"{PREFIX}/me", headers=FORGED).json()["user_id"] == "local"
    # With a token: the token's member, whatever the phone claims.
    assert loopback.get(f"{PREFIX}/me", headers={**FORGED, **T("member:ana")}).json()["user_id"] == "member:ana"
    # auth_required without a token or session: 401, the forged headers count for nothing.
    assert _app(True).get(f"{PREFIX}/me", headers=FORGED).status_code == 401


def test_members_cannot_see_each_others_records(store):
    seed(store)
    c = _app(False)
    rec = {"id": f"rec_{uuid.uuid4().hex[:16]}", "workflow_id": "equipment-inspection", "workflow_version": 1, "base_revision": 0,
           "answers": {"asset_id": "PMP-0042", "inspection_date": "2026-11-08", "condition": "good", "guard_ok": True},
           "created_local_at": "2026-11-08T10:00:00Z", "completed_at": "2026-11-08T10:05:00Z", "attachments": []}
    r = c.post(f"{PREFIX}/sync/records", headers=T("member:ana"),
               json={"idempotency_key": uuid.uuid4().hex, "device_id": "dev-1", "record": rec})
    assert r.status_code == 200, r.text
    assert store.get_record("member:ana", rec["id"])["id"] == rec["id"]
    with pytest.raises(db.NotFound):
        store.get_record("member:ben", rec["id"])
    with pytest.raises(db.NotFound):
        store.get_record("local", rec["id"])
    assert c.get(f"{PREFIX}/status", headers=T("member:ana")).json()["store"] != c.get(f"{PREFIX}/status", headers=T("member:ben")).json()["store"]


def test_origin_is_not_observed_when_public_url_is_set(store, monkeypatch):
    from fieldkit_core import runtime as rt
    c = _app(False)
    proxied = {"Host": "127.0.0.1:9119", "X-Forwarded-Proto": "https", "X-Forwarded-Host": "attacker.example"}
    monkeypatch.setenv("FIELDKIT_PUBLIC_URL", "https://door.example/field/ws1")
    for h in (proxied, {"Host": "attacker.example", "X-Forwarded-Proto": "https"}):
        assert c.get(f"{PREFIX}/status", headers={**h, **T("member:ana")}).status_code == 200
    assert rt.observed_url() == ""
    assert rt.public_url() == "https://door.example/field/ws1"
    # Without the setting the original detection still works (the upstream behaviour, see test_api.py).
    monkeypatch.delenv("FIELDKIT_PUBLIC_URL")
    c.get(f"{PREFIX}/status", headers={"Host": "fk.example.org", "X-Forwarded-Proto": "https"})
    assert rt.observed_url() == "https://fk.example.org"
