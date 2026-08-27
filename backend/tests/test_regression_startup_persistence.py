"""
Regression tests for startup safety and persistence reporting.

The reported symptom -- "registration works, existing users cannot log in
again" -- has two possible causes, and they need different fixes:

  (a) something on the startup path overwrites or shadows real accounts, or
  (b) the database file itself is gone, because DATA_DIR is ephemeral.

(a) is a code bug and is what these tests pin shut: the full boot sequence is
run repeatedly against a database containing real users, and nothing may change.
(b) is an infrastructure limitation that code cannot fix, so the tests instead
assert that the app REPORTS it rather than letting it look like an auth bug.
"""

import pytest

import backend.database as db
from backend.auth import hash_password, verify_password


USER = {
    "username": "RealUser",
    "password": "real-password-123",
    "email": "real.user@gmail.com",
    "full_name": "Real User",
}


def snapshot_users():
    """Every user row, ordered, as plain dicts -- comparable across boots."""
    with db.get_db_connection() as conn:
        rows = conn.execute("SELECT * FROM users ORDER BY id;").fetchall()
    return [dict(row) for row in rows]


def run_startup_sequence():
    """
    Everything that touches the database when the process boots.

    init_db() runs on import of backend.database; migrate_metadata_json() runs
    on import of backend.main; the demo seed runs from the startup event.
    """
    from backend.demo_seed import _demo_user_id
    from backend.main import migrate_metadata_json

    db.init_db()
    migrate_metadata_json()
    _demo_user_id(db, hash_password)


@pytest.fixture
def registered_user(client):
    resp = client.post("/api/auth/register", json=USER)
    assert resp.status_code == 200, resp.text
    return db.get_user_by_username(USER["username"])


# ---------------------------------------------------------------------------
# Startup must never mutate existing accounts
# ---------------------------------------------------------------------------

def test_startup_leaves_existing_users_byte_identical(client, registered_user):
    """
    Startup may ADD the bootstrap accounts it needs ('admin', the demo user) on
    a first boot. What it must never do is modify a row that already exists --
    that is the difference between seeding and clobbering.
    """
    before = {row["id"]: row for row in snapshot_users()}

    run_startup_sequence()

    after = {row["id"]: row for row in snapshot_users()}
    for user_id, original in before.items():
        assert user_id in after, f"startup deleted user {user_id}"
        assert after[user_id] == original, f"startup modified user {user_id}"


def test_repeated_startups_are_idempotent(client, registered_user):
    run_startup_sequence()
    after_first = snapshot_users()

    for _ in range(3):
        run_startup_sequence()

    assert snapshot_users() == after_first


def test_password_hash_is_untouched_by_startup(client, registered_user):
    original_hash = registered_user["password_hash"]
    run_startup_sequence()

    reloaded = db.get_user_by_username(USER["username"])
    assert reloaded["password_hash"] == original_hash
    assert verify_password(USER["password"], reloaded["password_hash"])


def test_login_still_works_after_repeated_startups(client, registered_user):
    for _ in range(3):
        run_startup_sequence()

    resp = client.post("/api/auth/login", json={
        "username": USER["username"], "password": USER["password"],
    })
    assert resp.status_code == 200
    assert resp.json()["username"] == USER["username"]


def test_startup_does_not_create_duplicate_accounts(client, registered_user):
    run_startup_sequence()
    run_startup_sequence()

    with db.get_db_connection() as conn:
        count = conn.execute(
            "SELECT COUNT(*) AS n FROM users WHERE username = ? COLLATE NOCASE;",
            (USER["username"],),
        ).fetchone()["n"]
    assert count == 1


def test_a_user_named_admin_is_not_replaced_by_the_bootstrap_account(client):
    """
    The startup migration creates an 'admin' record to own legacy documents. If
    a real person registered that username, their account must win.
    """
    resp = client.post("/api/auth/register", json={
        "username": "admin2",
        "password": "chosen-by-a-person",
        "email": "admin2@gmail.com",
        "full_name": "Real Admin",
    })
    assert resp.status_code == 200

    # 'admin' already exists from the test fixture; re-running must not touch it.
    admin_before = db.get_user_by_username("admin")
    run_startup_sequence()
    assert db.get_user_by_username("admin") == admin_before


def test_create_user_refuses_to_overwrite_an_existing_id(client, registered_user):
    """
    create_user is a plain INSERT, never an upsert. A colliding id or username
    must fail closed rather than replacing the row.
    """
    original = db.get_user_by_username(USER["username"])

    created = db.create_user(
        original["id"], "someone-else", hash_password("different"), "x@gmail.com", "X"
    )
    assert created is False
    assert db.get_user_by_id(original["id"]) == original


def test_init_db_is_safe_to_run_against_a_populated_database(client, registered_user):
    """Schema migrations use IF NOT EXISTS / guarded ALTERs, so re-running is a no-op."""
    before = snapshot_users()
    for _ in range(3):
        db.init_db()
    assert snapshot_users() == before


# ---------------------------------------------------------------------------
# The limitation must be visible, not silent
# ---------------------------------------------------------------------------

def test_health_reports_where_durable_state_lives(client):
    storage = client.get("/api/health").json()["storage"]
    assert storage["backend"] == "sqlite"
    assert storage["data_dir"]
    assert "database_existed_at_boot" in storage


def test_health_flags_an_attached_database_that_is_not_being_used(client):
    """
    Attaching Postgres on Render sets DATABASE_URL automatically. This build
    cannot use it, and silently ignoring it would let an operator believe their
    data is being persisted somewhere it is not.
    """
    import backend.main as main_module

    original = main_module.DATABASE_URL
    try:
        main_module.DATABASE_URL = "postgresql://user:pw@host/db"
        storage = client.get("/api/health").json()["storage"]
        assert storage["database_url_present_but_unused"] is True
    finally:
        main_module.DATABASE_URL = original

    # And it is not flagged when nothing is attached.
    assert client.get("/api/health").json()["storage"]["database_url_present_but_unused"] is False
