import pytest
from datetime import timedelta
import jwt
from backend.auth import (
    hash_password, verify_password, create_access_token, 
    decode_access_token, SECRET_KEY, ALGORITHM
)

def test_hash_password():
    pwd = "my_secure_password"
    hashed = hash_password(pwd)
    assert hashed != pwd
    assert hashed.startswith("$2b$") # standard bcrypt prefix

def test_verify_password():
    pwd = "secure_password"
    hashed = hash_password(pwd)
    assert verify_password(pwd, hashed) is True
    assert verify_password("wrong_password", hashed) is False

def test_create_access_token():
    data = {"sub": "user_123", "role": "admin"}
    token = create_access_token(data)
    assert isinstance(token, str)
    assert len(token) > 0
    
    # decode directly to check content
    decoded = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    assert decoded["sub"] == "user_123"
    assert decoded["role"] == "admin"
    assert "exp" in decoded

def test_decode_access_token_valid():
    data = {"sub": "user_456"}
    token = create_access_token(data)
    decoded = decode_access_token(token)
    assert decoded is not None
    assert decoded["sub"] == "user_456"

def test_decode_access_token_expired():
    # create a token that expired 10 minutes ago
    data = {"sub": "user_789"}
    expires = timedelta(minutes=-10)
    token = create_access_token(data, expires_delta=expires)
    decoded = decode_access_token(token)
    assert decoded is None

def test_decode_access_token_invalid():
    # try to decode a completely invalid string
    assert decode_access_token("invalid.token.string") is None
    
    # try to decode a token signed with a different key
    token = jwt.encode({"sub": "user_abc"}, "wrong_secret_key", algorithm=ALGORITHM)
    assert decode_access_token(token) is None
