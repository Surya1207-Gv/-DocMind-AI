import pytest

def test_register_success(client):
    payload = {
        "username": "newuser",
        "password": "securepwd",
        "email": "newuser@gmail.com",
        "full_name": "New User"
    }
    resp = client.post("/api/auth/register", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "access_token" in data
    assert data["username"] == "newuser"
    assert data["email"] == "newuser@gmail.com"

def test_register_duplicate_username(client):
    payload = {
        "username": "admin", # Already exists from conftest clean_db
        "password": "somepassword",
        "email": "admin2@gmail.com",
        "full_name": "Admin Two"
    }
    resp = client.post("/api/auth/register", json=payload)
    assert resp.status_code == 400
    assert "taken" in resp.json()["detail"].lower()

def test_register_invalid_email(client):
    payload = {
        "username": "newuser2",
        "password": "securepwd",
        "email": "invalid-email-format", # Malformed email without @ or domain
        "full_name": "New User Two"
    }
    resp = client.post("/api/auth/register", json=payload)
    assert resp.status_code == 400
    assert "invalid email" in resp.json()["detail"].lower()


def test_register_short_username(client):
    payload = {
        "username": "ab", # Less than 3 chars
        "password": "securepwd",
        "email": "ab@gmail.com",
        "full_name": "Ab Cd"
    }
    resp = client.post("/api/auth/register", json=payload)
    assert resp.status_code == 400
    assert "username must be" in resp.json()["detail"].lower()

def test_login_success(client):
    payload = {
        "username": "admin",
        "password": "admin123"
    }
    resp = client.post("/api/auth/login", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "access_token" in data
    assert data["username"] == "admin"

def test_login_failure(client):
    payload = {
        "username": "admin",
        "password": "wrong_password"
    }
    resp = client.post("/api/auth/login", json=payload)
    assert resp.status_code == 401
    assert "invalid username or password" in resp.json()["detail"].lower()

def test_update_profile_success(client, auth_headers):
    payload = {
        "username": "admin_updated",
        "email": "admin_updated@gmail.com",
        "full_name": "Admin Updated",
        "password": "newadminpassword"
    }
    resp = client.put("/api/users/me", json=payload, headers=auth_headers)
    assert resp.status_code == 200
    data = resp.json()
    assert data["username"] == "admin_updated"
    assert data["email"] == "admin_updated@gmail.com"

def test_update_profile_unauthorized(client):
    payload = {
        "username": "hacker",
        "email": "hacker@gmail.com",
    }
    resp = client.put("/api/users/me", json=payload)
    assert resp.status_code in (401, 403)
