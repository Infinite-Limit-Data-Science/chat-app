import os
from fastapi import APIRouter, Request, status, logger
from fastapi.responses import FileResponse

router = APIRouter(tags=["default"])


@router.get("/{path:path}", status_code=status.HTTP_404_NOT_FOUND)
async def catch_all(path: str):
    """Passthrough to ui to handle non-api prefixed routes"""
    index_file_path = os.path.abspath(
        os.path.join(os.getcwd(), "ui", "dist", "index.html")
    )
    return FileResponse(index_file_path)
