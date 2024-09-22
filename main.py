import os
import logging
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from clients.mongo_strategy import mongo_instance
from routes.home import router as home_router
from routes.conversations import router as conversations_router
from routes.messages import router as messages_router
from routes.settings import router as settings_router
from routes.default import router as default_router
from routes.streamingbots import router as streamingbot_router

load_dotenv()

logging.basicConfig(level=logging.DEBUG)

@asynccontextmanager
async def lifespan(_: FastAPI):
    try:
        await mongo_instance.connect()
    except Exception as e:
        msg = 'MongoDB connection error'
        logging.critical(f'{msg} {e}')
        raise RuntimeError(msg)

    logging.info(f'Database connection established')
    yield
    await mongo_instance.close()

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
app.include_router(default_router)
app.include_router(streamingbot_router)