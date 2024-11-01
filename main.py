import os
import logging
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from clients.mongo_strategy import mongo_instance as database_instance
from routes.home import router as home_router
from routes.conversations import router as conversations_router
from routes.messages import router as messages_router
from routes.settings import router as settings_router
from routes.default import router as default_router
from starlette.middleware.base import BaseHTTPMiddleware

load_dotenv()

logging.basicConfig(level=logging.WARNING)

@asynccontextmanager
async def lifespan(_: FastAPI):
    try:
        await database_instance.connect()
    except Exception as e:
        msg = 'Database connection error'
        logging.critical(f'{msg} {e}')
        raise RuntimeError(msg)

    logging.info(f'Database connection established')
    yield
    await database_instance.close()

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

class MultiAuthorizationMiddleware(BaseHTTPMiddleware):
    """Middleware to capture only the first valid Bearer token Authorization header, removing 'undefined' headers."""
    async def dispatch(self, request: Request, call_next):
        import re
        bearer_pattern = re.compile(r"^Bearer\s+.+")
        headers = dict(request.scope['headers'])
        valid_auth_header = next(
            (auth for auth in [val.decode('utf-8') for key, val in headers.items() if key == b'authorization']
            if bearer_pattern.match(auth) and auth.lower() != "undefined"),
            None,
        )

        if valid_auth_header:
            headers[b'authorization'] = valid_auth_header.encode()
            request.scope['headers'] = [(k, v) for k, v in headers.items()]
        else:
            request.scope['headers'] = [(k, v) for k, v in headers.items() if v != b'undefined']

        response = await call_next(request)
        return response

app.add_middleware(MultiAuthorizationMiddleware)

app.mount('/ui/assets', StaticFiles(directory='ui/dist/assets'), name='assets')

@app.get('/')
async def serve_root():
    return FileResponse('ui/dist/index.html')

@app.get('/health')
def health_check():
    return { 'message': 'ok' }

app.include_router(home_router, prefix='/api')
app.include_router(conversations_router, prefix='/api')
app.include_router(messages_router, prefix='/api')
app.include_router(settings_router, prefix='/api')
app.include_router(default_router)