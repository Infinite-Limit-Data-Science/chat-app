import os
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from .clients.mongo_strategy import mongo_instance as database_instance
from .routes.home import router as home_router
from .routes.conversations import router as conversations_router
from .routes.messages import router as messages_router
from .routes.settings import router as settings_router
from .routes.default import router as default_router
from .middleware import (
    MultiAuthorizationMiddleware, AddAuthorizationHeaderMiddleware)

load_dotenv()

@asynccontextmanager
async def lifespan(_: FastAPI):
    try:
        from pymongo import ASCENDING, DESCENDING
        await database_instance.connect()
        db = database_instance.get_database()
 
        await db.collection.create_index(
            [
                ('type', ASCENDING),
                ('content', ASCENDING),
                ('conversation_id', ASCENDING),
            ],
            name='type_content_conversation_id_index'
        )

        await db.collection.create_index(
            [
                ('type', ASCENDING),
                ('conversation_id', ASCENDING),
                ('createdAt', DESCENDING),
            ],
            name='type_conversation_id_createdAt_index'
        )
        
    except Exception as e:
        raise RuntimeError(f'Database connection error {e}')

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

app.add_middleware(MultiAuthorizationMiddleware)

class CustomStaticFiles(StaticFiles):
    async def get_response(self, path: str, scope):
        response = await super().get_response(path, scope)

        if path.endswith('.js'):
            response.headers['Content-Type'] = 'application/javascript'
        elif path.endswith('.svg'):
            response.headers['Content-Type'] = 'image/svg+xml'
        
        return response

app.mount('/assets', CustomStaticFiles(directory='ui/dist/assets'), name='assets')

if os.getenv('IS_LOCAL') == 'true' and os.getenv('JWT_LOOKUP') == 'true':
    app.add_middleware(AddAuthorizationHeaderMiddleware)

@app.get('/')
async def serve_root():
    return FileResponse('ui/dist/index.html')

@app.get('/health')
def health_check():
    return { 'message': 'ok' }

@app.middleware('http')
async def api_route_middleware(request: Request, call_next):
    if request.scope['path'].startswith('/api'):
        return await call_next(request)

    return await serve_static_file(request)

async def serve_static_file(request: Request):
    path = request.scope['path']
    file_path = f'ui/dist{path}'
    if os.path.exists(file_path) and file_path != 'ui/dist/':
        return FileResponse(file_path)
    else:
        return FileResponse(file_path)

_prefix = '/api'
app.include_router(home_router, prefix=_prefix)
app.include_router(conversations_router, prefix=_prefix)
app.include_router(messages_router, prefix=_prefix)
app.include_router(settings_router, prefix=_prefix)
app.include_router(default_router)