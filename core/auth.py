"""Registration and login with salted PBKDF2-SHA256 password hashes (standard library only)."""
import hashlib
import hmac
import re
import secrets

from core import db

ITERATIONS = 200_000
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    h = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, ITERATIONS)
    return f"pbkdf2_sha256${ITERATIONS}${salt.hex()}${h.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        algo, iters, salt_hex, hash_hex = stored.split("$")
    except ValueError:
        return False
    if algo != "pbkdf2_sha256":
        return False
    h = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), bytes.fromhex(salt_hex), int(iters))
    return hmac.compare_digest(h.hex(), hash_hex)


def register(name: str, email: str, password: str, city: str = None) -> tuple:
    """Returns (user dict, None) or (None, error message)."""
    name, email = (name or "").strip(), (email or "").strip().lower()
    if not name:
        return None, "Please enter your name."
    if not EMAIL_RE.match(email):
        return None, "Please enter a valid email address."
    if len(password or "") < 8:
        return None, "Password must be at least 8 characters."
    if db.get_user_by_email(email):
        return None, "An account with this email already exists."
    uid = db.create_user(name, email, hash_password(password), city=(city or "").strip() or None)
    return db.get_user(uid), None


def login(email: str, password: str) -> dict | None:
    u = db.get_user_by_email(email or "")
    if u and verify_password(password or "", u["password_hash"]):
        return u
    return None
