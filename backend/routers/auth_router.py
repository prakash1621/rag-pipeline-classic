"""
Auth router — login and token refresh endpoints.

Requirements: 1.1, 1.2, 1.4, 1.5
"""

import jwt
from fastapi import APIRouter, HTTPException, status

from backend.auth import (
    create_access_token,
    create_refresh_token,
    decode_token,
    load_users,
    verify_password,
)
from backend.models import (
    AccessTokenResponse,
    LoginRequest,
    RefreshRequest,
    TokenResponse,
)

router = APIRouter(prefix="/api/auth", tags=["auth"])


@router.post("/login", response_model=TokenResponse)
async def login(body: LoginRequest) -> TokenResponse:
    """Authenticate user and return access + refresh tokens."""
    users = load_users()

    user = users.get(body.username)
    if user is None or not verify_password(body.password, user["password_hash"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password",
        )

    payload = {"sub": body.username, "role": user["role"]}
    access_token = create_access_token(payload)
    refresh_token = create_refresh_token(payload)

    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        role=user["role"],
    )


@router.post("/refresh", response_model=AccessTokenResponse)
async def refresh(body: RefreshRequest) -> AccessTokenResponse:
    """Exchange a valid refresh token for a new access token."""
    try:
        token_payload = decode_token(body.refresh_token)
    except (jwt.ExpiredSignatureError, jwt.InvalidTokenError):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired refresh token",
        )

    if token_payload.get("type") != "refresh":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired refresh token",
        )

    new_access_token = create_access_token(
        {"sub": token_payload["sub"], "role": token_payload["role"]}
    )
    return AccessTokenResponse(access_token=new_access_token)
