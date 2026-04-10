"""
Document management router — list, upload, and delete knowledge-base files.

Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6
"""

import os

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, status

from app.config import KB_PATH
from backend.dependencies import require_admin
from backend.models import DocumentListResponse

router = APIRouter(prefix="/api", tags=["documents"])

ALLOWED_EXTENSIONS = {"pdf", "docx", "html", "txt", "md"}


@router.get("/documents", response_model=DocumentListResponse)
async def list_documents(_user: dict = Depends(require_admin)):
    """Scan the knowledge-base directory and return documents grouped by category."""
    documents: dict[str, list[str]] = {}

    if not os.path.isdir(KB_PATH):
        return DocumentListResponse(documents=documents)

    for entry in sorted(os.listdir(KB_PATH)):
        category_path = os.path.join(KB_PATH, entry)
        if os.path.isdir(category_path):
            files = sorted(
                f for f in os.listdir(category_path)
                if os.path.isfile(os.path.join(category_path, f))
            )
            documents[entry] = files

    return DocumentListResponse(documents=documents)


@router.post("/documents/upload")
async def upload_document(
    file: UploadFile = File(...),
    category: str = Form(...),
    _user: dict = Depends(require_admin),
):
    """Upload a document to the knowledge-base under the given category."""
    filename = file.filename or ""
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unsupported file format. Allowed: pdf, docx, html, txt, md",
        )

    category_dir = os.path.join(KB_PATH, category)
    os.makedirs(category_dir, exist_ok=True)

    file_path = os.path.join(category_dir, filename)
    content = await file.read()
    with open(file_path, "wb") as f:
        f.write(content)

    return {"filename": filename, "category": category}


@router.delete("/documents/{category}/{filename}")
async def delete_document(
    category: str,
    filename: str,
    _user: dict = Depends(require_admin),
):
    """Delete a document from the knowledge-base."""
    file_path = os.path.join(KB_PATH, category, filename)

    if not os.path.isfile(file_path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found",
        )

    os.remove(file_path)
    return {"detail": "Document deleted"}
