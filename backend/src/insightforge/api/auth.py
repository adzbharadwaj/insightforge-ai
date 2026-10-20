"""Authentication: bearer JWTs for users, and signed read-only share links for reports.

Every API query filters by the token's user id. Share links are separate tokens with a
`report:read` scope, a 7-day expiry and a `jti` that can be revoked (denylist table).
"""

import datetime as dt
import uuid
from typing import Annotated

import jwt
from fastapi import Depends, HTTPException, Query, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from insightforge.config import Settings, get_settings, require_secret
from insightforge.store.models import RevokedShareLink

ALGORITHM = "HS256"
USER_SCOPE = "user"
SHARE_SCOPE = "report:read"
_bearer = HTTPBearer(auto_error=False)


def _secret(settings: Settings) -> str:
    return require_secret(settings.jwt_secret, "JWT_SECRET")


def issue_access_token(user_id: uuid.UUID, settings: Settings | None = None) -> str:
    s = settings or get_settings()
    now = dt.datetime.now(dt.UTC)
    claims = {"sub": str(user_id), "scope": USER_SCOPE, "iat": now,
              "exp": now + dt.timedelta(hours=s.access_token_ttl_hours)}
    return jwt.encode(claims, _secret(s), algorithm=ALGORITHM)


def _decode(token: str, scope: str, settings: Settings) -> dict:
    try:
        claims = jwt.decode(token, _secret(settings), algorithms=[ALGORITHM], options={"require": ["exp", "sub"]})
    except jwt.ExpiredSignatureError as exc:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "token expired") from exc
    except jwt.InvalidTokenError as exc:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid token") from exc
    if claims.get("scope") != scope:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "wrong token scope")
    return claims


async def current_user(
    request: Request,
    creds: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
    token: Annotated[str | None, Query(description="For EventSource, which cannot send headers")] = None,
) -> uuid.UUID:
    raw = creds.credentials if creds else token
    if not raw:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "missing bearer token")
    return uuid.UUID(_decode(raw, USER_SCOPE, get_settings())["sub"])


CurrentUser = Annotated[uuid.UUID, Depends(current_user)]


# --- share links ------------------------------------------------------------------------


def issue_share_token(report_id: uuid.UUID, settings: Settings | None = None) -> tuple[str, str, dt.datetime]:
    s = settings or get_settings()
    jti = uuid.uuid4().hex
    expires = dt.datetime.now(dt.UTC) + dt.timedelta(days=s.share_link_ttl_days)
    token = jwt.encode({"sub": str(report_id), "scope": SHARE_SCOPE, "jti": jti, "exp": expires},
                       _secret(s), algorithm=ALGORITHM)
    return token, jti, expires


async def verify_share_token(token: str, sessionmaker: async_sessionmaker[AsyncSession]) -> uuid.UUID:
    claims = _decode(token, SHARE_SCOPE, get_settings())
    async with sessionmaker() as session:
        if await session.get(RevokedShareLink, claims.get("jti", "")):
            raise HTTPException(status.HTTP_410_GONE, "this share link was revoked")
    return uuid.UUID(claims["sub"])

