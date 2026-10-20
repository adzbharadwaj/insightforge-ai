"""Dev login. Production deployments get tokens from their identity provider instead."""

from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel, Field
from sqlalchemy import select

from insightforge.api.auth import issue_access_token
from insightforge.config import get_settings
from insightforge.store.models import User

router = APIRouter(prefix="/auth", tags=["auth"])


class DevLogin(BaseModel):
    # Format check only: dev identities like the CLI's dev@insightforge.local use reserved domains that a
    # deliverability validator rejects. Real identities come from the identity provider.
    email: str = Field(pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$", max_length=320)


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user_id: str


@router.post("/dev-login")
async def dev_login(request: Request, body: DevLogin) -> Token:
    if not get_settings().allow_dev_login:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "not found")
    async with request.app.state.sessionmaker() as session:
        user = await session.scalar(select(User).where(User.email == body.email.lower()))
        if user is None:
            user = User(email=body.email.lower())
            session.add(user)
            await session.commit()
    return Token(access_token=issue_access_token(user.id), user_id=str(user.id))
