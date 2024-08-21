import os
import logging
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import FastAPI, Request, Response, status, logger
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from chat_client import ChatClient as chat_client
from routes.home import router as home_router
from routes.conversations import router as conversations_router
from routes.messages import router as messages_router
from routes.settings import router as settings_router
from exceptions.exception_messages import OUT_404

load_dotenv()

logging.basicConfig(level=logging.DEBUG)

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        client = chat_client.instance()
        await client.connect(os.environ['MONGODB_URL'])
    except Exception as e:
        msg = 'MongoDB connection error'
        logging.critical(f'{msg} {e}')
        raise RuntimeError(msg)

    logging.info(f'MongoDB connection established')
    yield
    client.close()
    # Load ML model stuff

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.get('/health')
def health_check():
    return { 'message': 'ok' }

app.include_router(home_router)
app.include_router(conversations_router)
app.include_router(messages_router)
app.include_router(settings_router)

@app.get("/{path:path}", status_code=status.HTTP_404_NOT_FOUND)
async def catch_all(path: str, request: Request, response: Response):
    logger.logging.warning(f'WARNING: user entered invalid path {path}')
    accept_header = request.headers.get("Accept")
    if 'text/html' in accept_header:
        response = HTMLResponse(content=OUT_404, media_type='text/html')
        response.set_cookie(key='invalid_target', value='Target is invalid')
        return response
    else:
        return { 'error': f'path {path} does not exist' }