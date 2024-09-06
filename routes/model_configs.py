from fastapi import APIRouter, Request, status, Query, Body, Depends
from auth.bearer_authentication import get_current_user
from repositories.base_mongo_repository import base_mongo_factory as factory
from models.model_config import (
    ModelConfigEndpointSchema,
    ModelConfigPipelineSchema,
    ModelConfigIdSchema,
    UpdateModelConfigSchema,
    ModelConfigEndpointCollectionSchema,
    ModelConfigPipelineSchemaCollectionSchema,
    ModelConfig,
)

ModelConfigRepo = factory(ModelConfig)

router = APIRouter(
    prefix='/model_configs', 
    tags=['model_config'],
    dependencies=[Depends(get_current_user)]
)
@router.post(
    '/endpoint',
    response_description="Add new Model Config Endpoint",
    response_model=ModelConfigIdSchema,
    status_code=status.HTTP_201_CREATED,
    response_model_by_alias=False,
)
async def create_model_config_endpoint(request: Request, model_config_schema: ModelConfigEndpointSchema = Body(...)):
    """Insert new model config endpoint record in configured database, returning resource created"""
    if (
        created_model_config_id := await ModelConfigRepo.create(request.state.uuid, model_config_schema)
    ) is not None:
        return { "_id": created_model_config_id }
    return {'error': f'Model Config not created'}, 400

@router.post(
    '/',
    response_description="Add new Model Config Pipeline",
    response_model=ModelConfigIdSchema,
    status_code=status.HTTP_201_CREATED,
    response_model_by_alias=False,
)
async def create_model_config_pipeline(request: Request, setting_schema: ModelConfigPipelineSchema = Body(...)):
    """Insert new model config pipeline record in configured database, returning resource created"""
    if (
        created_model_config_id := await ModelConfigRepo.create(request.state.uuid, setting_schema)
    ) is not None:
        return { "_id": created_model_config_id }
    return {'error': f'Model Config not created'}, 400