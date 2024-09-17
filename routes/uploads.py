

def format_file_for_storage(uuid: str, conversation_id: str, id: str, filename: str):
    return f'files/{uuid}/conversations/{conversation_id}/messages/{id}/{filename}'

async def upload_file(request: Request, conversation_id: str, id: str, upload_file: UploadFile = File(...)) -> Optional[Dict[str,Any]]:
    uuid = request.state.uuid
    if (
        message_dict := await MessageRepo.find(uuid, conversation_id, id)
    ) is not None:
        path = format_file_for_storage(uuid, conversation_id, id, upload_file.filename)
        dir_path = os.path.dirname(path)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        with open(path, 'wb') as f:
            shutil.copyfileobj(upload_file.file, f)
        merged_files = ([] if message_dict['files'] is None else message_dict['files']) + [path]
        return await MessageRepo.update(uuid, conversation_id, id, UpdateMessageSchema(files=merged_files))

@router.post(
    '/{conversation_id}/message/{id}/upload',
    response_model=MessageSchema,
    response_description='Upload a file',
    tags=['message']
)
async def upload(request: Request, conversation_id: str, id: str, upload_file: UploadFile = File(...), message_dict: Optional[Dict[str,Any]] = Depends(upload_file)):
    if not message_dict:
        return {'error': f'Message {id} not found'}, 404
    file = format_file_for_storage(request.state.uuid, conversation_id, id, upload_file.filename)
    ingestor = DocumentIngestor(file, request.state.uuid, conversation_id, id)
    # TODO: make asynchronous
    embedded_ids = ingestor.ingest()
    logger.logging.warning(f'EMBEDDED IDs: {len(embedded_ids)}')
    return message_dict