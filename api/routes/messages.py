from typing import Optional, List, Union
import base64
from bson import ObjectId
from fastapi import (
    APIRouter, 
    status, 
    Request, 
    Form, 
    Depends, 
    File, 
    UploadFile,
)
from fastapi.exceptions import HTTPException
from fastapi.responses import StreamingResponse
from huggingface_hub.errors import HfHubHTTPError
from ..logger import logger
from ..models.mongo_schema import ObjectId
from ..auth.bearer_authentication import get_current_user
from .chats import chat
from .configs import (
    active_chat_bot_config,
    get_prompt_template,
    DEFAULT_PREPROMPT,
    DEFAULT_IMAGE_PROMPT,
)
from .uploads import ingest_files
from ..models.message import (
    Message,
    MessageSchema,
)
from ..repositories.base_mongo_repository import (
    base_mongo_factory as factory
)
from ..repositories.conversation_mongo_repository import (
    ConversationMongoRepository as ConversationRepo
)
from ..gwblue_chat_bot.chat_bot_config import ChatBotConfig

# TODO: extract to env var
_DATABASE_STRATEGY = 'mongodb'

MessageRepo = factory(Message)

router = APIRouter(
    prefix='/conversations', 
    tags=['conversation'],
    dependencies=[Depends(get_current_user)]
)

@router.post(
    '/{conversation_id}/message',
    response_description="Add new message",
    status_code=status.HTTP_201_CREATED,
    tags=['message']
)
async def create_message(
    request: Request,
    conversation_id: str,
    content: str = Form(...),
    upload_files: Optional[List[UploadFile]] = File(None),
    chat_bot_config: ChatBotConfig = Depends(active_chat_bot_config),
    system_prompt: str = Depends(get_prompt_template),
):
    logger.info(f'invoking message endpoint with content `{content}`')

    if _DATABASE_STRATEGY == 'mongodb':
        conversation_id = ObjectId(conversation_id)

    ingestible_files = []
    image_files = []
    if upload_files:
        for f in upload_files:
            if f.content_type in [
                'application/pdf',
                'application/msword',
                'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
                'application/vnd.openxmlformats-officedocument.presentationml.presentation',
                'text/plain',
            ]:
                ingestible_files.append(f)
            elif f.content_type.startswith('image/'):
                image_files.append(f)
            else:
                logger.warning(f'Unrecognized file type {f.filename} with content_type {f.content_type}')

    vectorstore_metadata = []
    filenames = []
    if ingestible_files:
        vectorstore_metadata, filenames = await ingest_files(
            files=ingestible_files,
            config=chat_bot_config,
            metadata={
                'uuid': request.state.uuid,
                'conversation_id': str(conversation_id)
            },
        )
        await ConversationRepo.update_one(
            conversation_id,
            _set={'filenames': filenames}
        )
    else:
        vectorstore_metadata = [
            {
                'uuid': request.state.uuid,
                'conversation_id': str(conversation_id),
            }            
        ]

    image_prompts = []
    for img_file in image_files:
        raw_bytes = await img_file.read()
        encoded = base64.b64encode(raw_bytes).decode('utf-8')
        subtype = img_file.content_type.split('/')[-1]
        image_url = f'data:image/{subtype};base64,{encoded}'
        image_prompts.append({'image_url': {'url': image_url}})

    base_system_prompt = DEFAULT_IMAGE_PROMPT if len(image_prompts) > 0 else DEFAULT_PREPROMPT
    system_prompt = system_prompt or base_system_prompt

    user_prompt_parts = [content]
    for img_obj in image_prompts:
        user_prompt_parts.append(img_obj)
    if image_prompts:
        user_prompt_parts.append('Please describe the image(s).')

    chat_bot_config.message_history.session_id = conversation_id

    try:
        llm_stream = await chat(
            system=system_prompt,
            input=user_prompt_parts,
            config=chat_bot_config,
            metadata=vectorstore_metadata,
        )
        return StreamingResponse(
            llm_stream(),
            media_type='text/event-stream',
            headers={'X-Accel-Buffering': 'no'}
        )
    except HfHubHTTPError as e:
        error_info = {
            'url': e.response.url,
            'status_code': e.response.status_code,
            'error_message': e.response.text,
            'error_type': type(e).__name__,
        }
        logger.warning(f'Request failed error_info {error_info}')
        raise HTTPException(status_code=e.response.status_code, detail=error_info)

@router.get(
    '/{conversation_id}/message/{id}',
    response_description="Get a single message",
    response_model=Union[MessageSchema, dict],
    response_model_by_alias=False,
    tags=['message']
)
async def get_message(request: Request, conversation_id: str, id: str):
    """Get message record from configured database by id"""
    if (
        message := await MessageRepo.find_one(id, options={'conversation_id': ObjectId(conversation_id) })
    ) is not None:
        return message
    
    return {}

# Note Conversational AI does not allow the edit of existing messages
@router.delete(
    '/{conversation_id}/message/{id}', 
    response_description='Delete a message',
    tags=['message']
)
async def delete_message(request: Request, conversation_id: str, id: str):
    """Remove a single message record from the database."""
    if (
        delete_count := await MessageRepo.delete(id, options={'conversation_id': ObjectId(conversation_id)})
    ) is not None:
        await ConversationRepo.remove_from_field(conversation_id, options={'message_ids': ObjectId(id) })
        return {'delete_count': delete_count} 
     
    return {'delete_count': 0}