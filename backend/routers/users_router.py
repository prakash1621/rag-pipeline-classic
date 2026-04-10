"""
Users router — user management endpoints (list, create, delete).

Requirements: 3.1, 3.2, 3.3, 3.4, 3.5
"""

from fastapi import APIRouter, Depends, HTTPException, status

from backend.auth import hash_password, load_users, save_users
from backend.dependencies import require_admin
from backend.models import CreateUserRequest, UserOut

router = APIRouter(prefix="/api", tags=["users"])


@router.get("/users", response_model=list[UserOut])
async def list_users(_admin: dict = Depends(require_admin)) -> list[UserOut]:
    """Return all users with username and role only (no password hashes)."""
    users = load_users()
    return [
        UserOut(username=username, role=info["role"])
        for username, info in users.items()
    ]


@router.post("/users", response_model=UserOut, status_code=status.HTTP_201_CREATED)
async def create_user(
    body: CreateUserRequest,
    _admin: dict = Depends(require_admin),
) -> UserOut:
    """Create a new user with a bcrypt-hashed password."""
    users = load_users()

    if body.username in users:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="User already exists",
        )

    users[body.username] = {
        "password_hash": hash_password(body.password),
        "role": body.role,
    }
    save_users(users)

    return UserOut(username=body.username, role=body.role)


@router.delete("/users/{username}")
async def delete_user(
    username: str,
    _admin: dict = Depends(require_admin),
) -> dict:
    """Remove a user from the store."""
    users = load_users()

    if username not in users:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )

    del users[username]
    save_users(users)

    return {"detail": "User deleted"}
