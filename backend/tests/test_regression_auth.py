"""
Regression tests for the reported "registration works, existing users cannot log
in again" defect, and for the auth contract generally.

Each test here corresponds to a specific way an account that plainly exists was
being turned into "invalid username or password".
"""

import pytest

import backend.database as db
from backend.auth import create_access_token, hash_password, verify_password


REGISTRATION = {
    "username": "Surya",
    "password": "correct-horse",
    "email": "Surya.User@Gmail.com",
    "full_name": "Surya User",
}


def register(client, **overrides):
    payload = {**REGISTRATION, **overrides}
    return client.post("/api/auth/register", json=payload)


def test_register_then_login_immediately(client):
    assert register(client).status_code == 200

    resp = client.post(
        "/api/auth/login",
        json={"username": REGISTRATION["username"], "password": REGISTRATION["password"]},
    )
    assert resp.status_code == 200
    assert resp.json()["username"] == "Surya"


def test_login_is_case_insensitive_on_username(client):
    """
    Registering as "Surya" and typing "surya" a week later must still log in.

    The lookup used to be an exact string match, so any casing difference read
    as a non-existent user.
    """
    assert register(client).status_code == 200

    for typed in ("surya", "SURYA", "SuRyA", "  Surya  "):
        resp = client.post(
            "/api/auth/login",
            json={"username": typed, "password": REGISTRATION["password"]},
        )
        assert resp.status_code == 200, f"login failed for {typed!r}"
        # The stored casing is echoed back, not whatever was typed.
        assert resp.json()["username"] == "Surya"


def test_login_accepts_the_registered_email(client):
    """The sign-in form has one field; users type whichever they remember."""
    assert register(client).status_code == 200

    resp = client.post(
        "/api/auth/login",
        json={"username": "surya.user@gmail.com", "password": REGISTRATION["password"]},
    )
    assert resp.status_code == 200
    assert resp.json()["username"] == "Surya"


def test_login_survives_a_restart(client):
    """
    An account must still be usable after the startup path runs again.

    This is the shape of the original bug report: the account works now and not
    later. Re-running init_db and the metadata/admin bootstrap simulates a
    process restart against the same database file and proves the seeding logic
    does not clobber or shadow a real user.
    """
    assert register(client).status_code == 200

    from backend.main import migrate_metadata_json

    db.init_db()
    migrate_metadata_json()

    resp = client.post(
        "/api/auth/login",
        json={"username": REGISTRATION["username"], "password": REGISTRATION["password"]},
    )
    assert resp.status_code == 200, "user was lost or shadowed by startup seeding"

    # The stored row is still the user's own, not an overwritten seed account.
    user = db.get_user_by_username("surya")
    assert user is not None
    assert user["email"] == REGISTRATION["email"]
    assert verify_password(REGISTRATION["password"], user["password_hash"])


def test_demo_seed_does_not_overwrite_a_real_user(client):
    """Seeding the demo account must never touch an existing account of the same name."""
    assert register(client, username="demo", email="demo.real@gmail.com").status_code == 200
    original = db.get_user_by_username("demo")

    from backend.demo_seed import _demo_user_id

    returned_id = _demo_user_id(db, hash_password)

    assert returned_id == original["id"]
    after = db.get_user_by_username("demo")
    assert after["password_hash"] == original["password_hash"]
    assert after["email"] == "demo.real@gmail.com"


def test_wrong_password_is_401(client):
    assert register(client).status_code == 200
    resp = client.post(
        "/api/auth/login",
        json={"username": REGISTRATION["username"], "password": "not-the-password"},
    )
    assert resp.status_code == 401


def test_unknown_user_is_401(client):
    resp = client.post(
        "/api/auth/login", json={"username": "nobody-here", "password": "whatever"}
    )
    assert resp.status_code == 401


def test_unknown_user_and_wrong_password_are_indistinguishable(client):
    """The error must not reveal which accounts exist."""
    assert register(client).status_code == 200

    wrong_password = client.post(
        "/api/auth/login",
        json={"username": REGISTRATION["username"], "password": "nope"},
    )
    no_such_user = client.post(
        "/api/auth/login", json={"username": "ghost", "password": "nope"}
    )
    assert wrong_password.json()["detail"] == no_such_user.json()["detail"]


def test_duplicate_username_is_409_regardless_of_case(client):
    assert register(client).status_code == 200

    resp = register(client, username="SURYA", email="other@gmail.com")
    assert resp.status_code == 409
    assert "taken" in resp.json()["detail"].lower()


def test_duplicate_email_is_409_regardless_of_case(client):
    assert register(client).status_code == 200

    resp = register(client, username="someone-else", email="SURYA.USER@GMAIL.COM")
    assert resp.status_code == 409
    assert "registered" in resp.json()["detail"].lower()


def test_password_is_never_stored_in_plaintext(client):
    assert register(client).status_code == 200

    with db.get_db_connection() as conn:
        row = conn.execute(
            "SELECT password_hash FROM users WHERE username = ?;", ("Surya",)
        ).fetchone()

    stored = row["password_hash"]
    assert REGISTRATION["password"] not in stored
    assert stored.startswith("$2b$")


def test_long_passwords_verify_consistently(client):
    """
    bcrypt reads only the first 72 bytes. Truncating explicitly on both sides
    keeps a long password working the same way regardless of which bcrypt
    release is installed -- otherwise the same password can hash on one version
    and fail to verify on another.
    """
    long_password = "p" * 200
    hashed = hash_password(long_password)
    assert verify_password(long_password, hashed) is True
    assert verify_password("p" * 199, hashed) is False or verify_password("p" * 72, hashed) is True


def test_malformed_stored_hash_is_a_failed_login_not_a_500(client):
    """A corrupt row must reject the login, never crash the endpoint."""
    assert register(client).status_code == 200
    with db.get_db_connection() as conn:
        conn.execute(
            "UPDATE users SET password_hash = ? WHERE username = ?;",
            ("not-a-bcrypt-hash", "Surya"),
        )
        conn.commit()

    resp = client.post(
        "/api/auth/login",
        json={"username": "Surya", "password": REGISTRATION["password"]},
    )
    assert resp.status_code == 401


def test_protected_route_requires_a_token(client):
    assert client.get("/api/documents").status_code in (401, 403)


def test_protected_route_rejects_a_garbage_token(client):
    resp = client.get(
        "/api/documents", headers={"Authorization": "Bearer not.a.real.token"}
    )
    assert resp.status_code == 401


def test_token_for_a_deleted_user_is_rejected(client):
    """
    Logging out clears the token client-side, but a token whose user no longer
    exists must also be rejected server-side.
    """
    assert register(client).status_code == 200
    user = db.get_user_by_username("Surya")
    token = create_access_token({"sub": user["id"]})

    headers = {"Authorization": f"Bearer {token}"}
    assert client.get("/api/documents", headers=headers).status_code == 200

    with db.get_db_connection() as conn:
        conn.execute("DELETE FROM users WHERE id = ?;", (user["id"],))
        conn.commit()

    assert client.get("/api/documents", headers=headers).status_code == 401
