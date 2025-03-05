import os
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
load_dotenv()
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path
from .clients.mongo_strategy import mongo_instance as history_store
from .routes.home import router as home_router
from .routes.conversations import router as conversations_router
from .routes.messages import router as messages_router
from .routes.settings import router as settings_router
from .routes.default import router as default_router
from .middleware import (
    MultiAuthorizationMiddleware, 
    AddAuthorizationHeaderMiddleware
)
from redis.client import Redis
from redis.connection import ConnectionPool
from pymongo import ASCENDING, DESCENDING

load_dotenv()

_REDIS_MAX_CONNECTIONS = 50
_REDIS_SOCKET_TIMEOUT = 30.0

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        # TODO: put this in a wrapper so we can swap
        # vector stores in the future
        redis_client = Redis.from_pool(
            ConnectionPool.from_url(
                os.environ['REDIS_URL'], 
                max_connections=_REDIS_MAX_CONNECTIONS,
                socket_timeout=_REDIS_SOCKET_TIMEOUT
            )
        )
        app.state.redis_client = redis_client

        await history_store.connect()
        db = history_store.get_database()
 
        cursor = db['messages'].list_indexes()
        existing_indexes = await cursor.to_list(length=None)
        existing_index_names = [ix['name'] for ix in existing_indexes]
        if 'type_content_conversation_id_index' not in existing_index_names:
            await db['messages'].create_index(
                [
                    ('type', ASCENDING),
                    ('content', ASCENDING),
                    ('conversation_id', ASCENDING),
                ],
                name='type_content_conversation_id_index'
            )

        if 'type_conversation_id_createdAt_index' not in existing_index_names:
            await db['messages'].create_index(
                [
                    ('type', ASCENDING),
                    ('conversation_id', ASCENDING),
                    ('createdAt', DESCENDING),
                ],
                name='type_conversation_id_createdAt_index'
            )
        
    except Exception as e:
        raise RuntimeError(f'Client connection error {e}')

    yield
    redis_client.close()
    await history_store.close()

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

app.add_middleware(MultiAuthorizationMiddleware)

class CustomStaticFiles(StaticFiles):
    async def get_response(self, path: str, scope):
        response = await super().get_response(path, scope)

        if path.endswith('.js'):
            response.headers['Content-Type'] = 'application/javascript'
        elif path.endswith('.svg'):
            response.headers['Content-Type'] = 'image/svg+xml'
        
        return response

path = Path(__file__).resolve().parent.parent / 'ui/dist/assets'
app.mount('/assets', CustomStaticFiles(directory=path), name='assets')

if os.getenv('IS_LOCAL') == 'true' and os.getenv('JWT_LOOKUP') == 'true':
    app.add_middleware(AddAuthorizationHeaderMiddleware)

@app.middleware('http')
async def api_route_middleware(request: Request, call_next):
    if request.scope['path'].startswith('/api'):
        return await call_next(request)
    
    if request.scope['path'].startswith('/health'):
        return Response(content='{"message": "ok"}', media_type="application/json")

    try:
        return await serve_static_file(request)
    except Exception as e:
        return Response(content=f"Error: {str(e)}", status_code=500)

async def serve_static_file(request: Request):
    path = request.scope['path']
    file_path = f'ui/dist{path}'
    if os.path.exists(file_path) and file_path != 'ui/dist/':
        return FileResponse(file_path)
    else:
        return FileResponse('ui/dist/index.html')

_prefix = '/api'
app.include_router(home_router, prefix=_prefix)
app.include_router(conversations_router, prefix=_prefix)
app.include_router(messages_router, prefix=_prefix)
app.include_router(settings_router, prefix=_prefix)
app.include_router(default_router)