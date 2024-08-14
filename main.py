import os
import logging
from contextlib import asynccontextmanager
from motor import motor_asyncio
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from chat_client import ChatClient as chat_client
from routes.home import router as home_router
from routes.conversations import router as conversations_router
from routes.messages import router as message_router

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
    allow_origins=["*"], # TODO: restrict
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.get('/health')
def health_check():
    return { 'message': 'ok' }

app.include_router(home_router)
app.include_router(conversations_router)
app.include_router(message_router)