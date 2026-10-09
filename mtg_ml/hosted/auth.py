"""Cloudflare Access: who is asking.

Access sits in front of the server (Cloudflare Tunnel; the origin listens on
loopback only) and signs every request it lets through with a JWT, sent in
the `Cf-Access-Jwt-Assertion` header (and the `CF_Authorization` cookie).
The server still verifies it, following
https://developers.cloudflare.com/cloudflare-one/access-controls/applications/http-apps/authorization-cookie/validating-json/ :

- RS256 signature against the team's keys at `<team>/cdn-cgi/access/certs`
  (cached; refetched on an unknown `kid`, at most once a minute, and hourly
  otherwise: Access rotates keys every six weeks with an overlap),
- `iss` equal to the team domain, `aud` containing the application's tag,
- `exp`, `nbf` and `iat` with a small leeway,
- the `email` claim, lower-cased, on the allowlist.

`fetch_certs` is injectable (tests sign tokens with a local key).
"""

from __future__ import annotations

import json
import threading
import time
import urllib.request

import jwt

HEADER = "Cf-Access-Jwt-Assertion"
COOKIE = "CF_Authorization"
LEEWAY = 30  # seconds of clock skew tolerated on exp / nbf / iat
REFRESH_MIN = 60  # an unknown kid refetches the keys at most this often
KEYS_TTL = 3600  # known keys are refetched this often anyway
MAX_KEYS = 32
MAX_CERTS_BYTES = 1 << 20


class AuthError(Exception):
    """Refused: `status` 401 (no or invalid credentials) or 403 (not allowed)."""

    def __init__(self, message: str, status: int = 401):
        super().__init__(message)
        self.status = status


def http_certs(team_domain: str):
    """The default fetcher: the team's JWKS document."""
    url = f"{team_domain}/cdn-cgi/access/certs"

    def fetch() -> dict:
        req = urllib.request.Request(url, headers={"Accept": "application/json", "User-Agent": "mtg-ml-hosted"})
        with urllib.request.urlopen(req, timeout=5) as r:
            return json.loads(r.read(MAX_CERTS_BYTES))

    return fetch


class AccessVerifier:
    def __init__(self, team_domain: str, aud: str, allowed_emails, fetch_certs=None, leeway: int = LEEWAY, clock=time.time):
        self.issuer = team_domain.rstrip("/")
        self.aud = aud
        self.allowed = frozenset(e.strip().lower() for e in allowed_emails)
        self.fetch_certs = fetch_certs or http_certs(self.issuer)
        self.leeway, self.clock = leeway, clock  # clock: paces key fetches (claims use the real time)
        self._keys: dict[str, object] = {}
        self._fetched = None  # clock time of the last fetch attempt
        self._lock = threading.Lock()

    def _refresh(self) -> None:
        self._fetched = self.clock()
        try:
            doc = self.fetch_certs()
            keys = {}
            for jwk in (doc.get("keys") or [])[:MAX_KEYS]:
                if jwk.get("kty") == "RSA" and jwk.get("kid"):
                    try:
                        keys[str(jwk["kid"])] = jwt.PyJWK(jwk, algorithm="RS256").key
                    except Exception:  # one malformed key does not spoil the rest
                        continue
        except Exception:
            return  # keep the keys we have; an unknown kid fails below
        if keys:
            self._keys = keys

    def key(self, kid: str):
        with self._lock:
            now = self.clock()
            stale = self._fetched is None or now - self._fetched >= KEYS_TTL
            unknown = kid not in self._keys and (self._fetched is None or now - self._fetched >= REFRESH_MIN)
            if stale or unknown:
                self._refresh()
            return self._keys.get(kid)

    def verify(self, token: str | None) -> str:
        """The verified, allowlisted email (lower-case), or AuthError."""
        if not token:
            raise AuthError("sign in through Cloudflare Access (no Access token)")
        try:
            header = jwt.get_unverified_header(token)
        except jwt.PyJWTError:
            raise AuthError("invalid Access token") from None
        if header.get("alg") != "RS256":
            raise AuthError("invalid Access token (algorithm)")
        kid = header.get("kid")
        key = self.key(str(kid)) if kid else None
        if key is None:
            raise AuthError("invalid Access token (unknown signing key)")
        try:
            claims = jwt.decode(token, key, algorithms=["RS256"], audience=self.aud, issuer=self.issuer, leeway=self.leeway,
                                options={"require": ["exp", "iat", "iss", "aud"], "verify_signature": True, "verify_exp": True, "verify_nbf": True,
                                         "verify_iat": True, "verify_aud": True, "verify_iss": True})
        except jwt.ExpiredSignatureError:
            raise AuthError("your Access session expired: reload the page") from None
        except jwt.PyJWTError as e:
            raise AuthError(f"invalid Access token ({type(e).__name__})") from None
        iat = claims.get("iat")
        if not isinstance(iat, (int, float)) or iat > time.time() + self.leeway:
            raise AuthError("invalid Access token (issued in the future)")
        email = claims.get("email")
        if not isinstance(email, str) or "@" not in email:
            raise AuthError("the Access token carries no email")
        email = email.strip().lower()
        if email not in self.allowed:
            raise AuthError("this account is not on the list of players", 403)
        return email


def request_token(headers, cookies) -> str | None:
    """The Access JWT of a request: the header, else the cookie."""
    return headers.get(HEADER) or cookies.get(COOKIE) or None
