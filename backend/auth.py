"""
Auth module — JWT creation/validation, bcrypt password hashing, YAML user store I/O.

Requirements: 1.1, 1.3, 1.4, 1.5, 1.6, 1.7, 13.1, 13.2, 13.3
"""

import os
import platform
from datetime import datetime, timedelta, timezone

import bcrypt
import jwt
import yaml

# ── Constants ───────────────────────────────────────────────
SECRET_KEY: str = os.environ.get("SECRET_KEY", "dev-secret-key-change-in-production")
ALGORITHM: str = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
REFRESH_TOKEN_EXPIRE_DAYS: int = 7

# Path to users.yaml in the rag-pipeline-classic/ directory
_BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_BACKEND_DIR)
USER_STORE_PATH: str = os.path.join(_PROJECT_ROOT, "users.yaml")


# ── Password hashing ───────────────────────────────────────

def hash_password(plain: str) -> str:
    """Hash a plaintext password using bcrypt. Returns the hash as a string."""
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(plain.encode("utf-8"), salt)
    return hashed.decode("utf-8")


def verify_password(plain: str, hashed: str) -> bool:
    """Verify a plaintext password against a bcrypt hash."""
    return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))


# ── JWT token helpers ───────────────────────────────────────

def create_access_token(data: dict, expires_delta: timedelta | None = None) -> str:
    """Create a JWT access token with HS256.

    The payload includes: sub, role, exp, type="access".
    Default expiry is ACCESS_TOKEN_EXPIRE_MINUTES (30 min).
    """
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (
        expires_delta if expires_delta is not None else timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    to_encode.update({"exp": expire, "type": "access"})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def create_refresh_token(data: dict) -> str:
    """Create a JWT refresh token with HS256.

    The payload includes: sub, role, exp, type="refresh".
    Default expiry is REFRESH_TOKEN_EXPIRE_DAYS (7 days).
    """
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    to_encode.update({"exp": expire, "type": "refresh"})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def decode_token(token: str) -> dict:
    """Decode a JWT token. Returns the payload dict.

    Raises:
        jwt.ExpiredSignatureError: if the token has expired.
        jwt.InvalidTokenError: if the token is otherwise invalid.
    """
    return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])


# ── YAML user store I/O ────────────────────────────────────

def load_users() -> dict:
    """Read users.yaml and return {username: {password_hash, role}}.

    Returns an empty dict if the file does not exist or is empty.
    """
    if not os.path.isfile(USER_STORE_PATH):
        return {}

    with open(USER_STORE_PATH, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)

    if not data or "users" not in data:
        return {}

    return data["users"]


def save_users(users: dict) -> None:
    """Write users dict to users.yaml with platform-aware file locking.

    The file format is:
        users:
          username:
            password_hash: "..."
            role: "admin" | "viewer"
    """
    data = {"users": users}
    yaml_content = yaml.dump(data, default_flow_style=False)

    if platform.system() == "Windows":
        _save_users_windows(yaml_content)
    else:
        _save_users_unix(yaml_content)


def _save_users_unix(yaml_content: str) -> None:
    """Write users.yaml with fcntl file locking (Unix/macOS)."""
    import fcntl

    with open(USER_STORE_PATH, "w", encoding="utf-8") as fh:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        try:
            fh.write(yaml_content)
        finally:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)


def _save_users_windows(yaml_content: str) -> None:
    """Write users.yaml with msvcrt file locking (Windows)."""
    import msvcrt

    with open(USER_STORE_PATH, "w", encoding="utf-8") as fh:
        msvcrt.locking(fh.fileno(), msvcrt.LK_LOCK, 1)
        try:
            fh.write(yaml_content)
        finally:
            try:
                msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
            except OSError:
                pass


def ensure_default_admin() -> None:
    """Create users.yaml with a default admin/admin account if the file is missing or empty."""
    users = load_users()
    if users:
        return  # Users already exist, nothing to do

    admin_hash = hash_password("admin")
    users = {
        "admin": {
            "password_hash": admin_hash,
            "role": "admin",
        }
    }
    save_users(users)
