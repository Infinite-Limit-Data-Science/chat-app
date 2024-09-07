from fastapi import APIRouter, Request, status, Query, Body, Depends
from auth.bearer_authentication import get_current_user
from repositories.base_mongo_repository import base_mongo_factory as factory
from models.model_config import (
    ModelConfigSchema,
    ModelConfigIdSchema,
    UpdateModelConfigSchema,
    ModelConfigCollectionSchema,
    ModelConfig,
)

ModelConfigRepo = factory(ModelConfig)

router = APIRouter(
    prefix='/model_configs', 
    tags=['model_config'],
    dependencies=[Depends(get_current_user)]
)

@router.get(
    '/',
    response_description='List all model configs',
    response_model=ModelConfigCollectionSchema,
    response_model_by_alias=False,
)
async def model_configs(request: Request, record_offset: int = Query(0, description='record offset', alias='offset'), record_limit: int = Query(20, description="record limit", alias='limit')):
    """List conversations by an offset and limit"""    
    return ModelConfigCollectionSchema(model_configs=await ModelConfigRepo.all(request.state.uuid, record_offset, record_limit))

@router.post(
    '/',
    response_description="Add new model config",
    response_model=ModelConfigIdSchema,
    status_code=status.HTTP_201_CREATED,
    response_model_by_alias=False,
)
async def create_model_config(request: Request, model_config_schema: ModelConfigSchema = Body(...)):
    """Insert new model config record in configured database, returning resource created"""
    if (
        created_model_config_id := await ModelConfigRepo.create(request.state.uuid, model_config_schema)
    ) is not None:
        return { "_id": created_model_config_id }
    return {'error': f'Model Config not created'}, 400

@router.get(
    '/{id}',
    response_description="Get a single model config",
    response_model=ModelConfigSchema,
    response_model_by_alias=False
)
async def get_model_config(request: Request, id: str):
    """Get model config record from configured database by id"""
    if (
        model_config := await ModelConfigRepo.find(request.state.uuid, id)
    ) is not None:
        return model_config
    return {'error': f'Model Config {id} not found'}, 404

@router.put(
    '/{id}',
    response_description="update a single model config",
    response_model=ModelConfigSchema,
    response_model_by_alias=False
)
async def update_model_config(request: Request, id: str, setting_schema: UpdateModelConfigSchema = Body(...)):
    """Get model config record from configured database by id"""
    if (
        updated_model_config := await ModelConfigRepo.update(request.state.uuid, id, setting_schema)
    ) is not None:
        return updated_model_config
    return {'error': f'Model Config {id} not found'}, 404

@router.delete(
    '/{id}', 
    response_description='Delete a model config',
)
async def delete_model_config(request: Request, id: str):
    """Remove a single model config record from the database."""
    if (
        deleted_model_config := await ModelConfigRepo.delete(request.state.uuid, id)
    ) is not None:
        return deleted_model_config  
    return { 'error': f'Model Config {id} not found'}, 404