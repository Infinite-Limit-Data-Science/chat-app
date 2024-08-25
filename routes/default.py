from fastapi import APIRouter, Request, status, logger
from fastapi.responses import HTMLResponse
from exceptions.exception_messages import OUT_404

router = APIRouter(
    tags=['default']
)

@router.get("/{path:path}", status_code=status.HTTP_404_NOT_FOUND)
async def catch_all(path: str, request: Request):
    logger.logging.warning(f'WARNING: user entered invalid path {path}')
    accept_header = request.headers.get("Accept")
    if 'text/html' in accept_header:
        response = HTMLResponse(content=OUT_404, media_type='text/html')
        response.set_cookie(key='invalid_target', value='Target is invalid')
        return response
    else:
        return { 'error': f'path {path} does not exist' }