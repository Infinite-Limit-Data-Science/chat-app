from datetime import datetime
from typing import TypeAlias, Optional, Annotated
from pydantic import BaseModel, Field, BeforeValidator, AfterValidator, model_serializer
from bson import ObjectId

PyObjectId = Annotated[str, BeforeValidator(str), AfterValidator(lambda v: ObjectId(v))]

ChatSchema: TypeAlias = BaseModel

def jsonify_object_ids(data):
    """
        FastAPI internally invokes model_dump_json on a Pydantic model when serializing to json for http 
        response. When request comes into route, json is deserialized into Pydantic objects in FastApi. 
        model_dump_json's behavior is different from the behavior of model_dump which just converts pydantic 
        model to dictionary. In most situations, the two work well together. However, in mongo schemas, ObjectId
        is not a json serializable object. Yet, we need ObjectIds when writing to a mongo database.

        When storing complex data structures, such as mongodb documents, which have subdocuments, which in
        turn have more subdocuments, all of which have ObjectIds, in mongo, these ObjectIds cannot be strings.
        Without using model_dump and AfterValidator, then it will require a lot more code to get the strings
        back to ObjectIds. Hence, the AfterValidator in PyObjectId to preserve the ObjectIds for complex
        nested documents. Concurrently, this has side effect of making model_dump_json return ObjectIds too.
        Consequenty, this modifier function is only invoked when attempts are made to jsonify data to ensure
        strings are always returned (and thus never ObjectIds).

        BeforeValidator is also required.
    """
    if isinstance(data, dict):
        return {k: jsonify_object_ids(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [jsonify_object_ids(item) for item in data]
    elif isinstance(data, ObjectId):
        return str(data)
    else:
        return data

class PrimaryKeyMixinSchema(ChatSchema):
    id: Optional[PyObjectId] = Field(alias='_id', description='bson object id', default_factory=ObjectId)

    @model_serializer(mode='plain', when_used='json')
    def serialize_object_id(self) -> dict:
        dict = self.model_dump()
        json_compatible_data = jsonify_object_ids(dict)
        return json_compatible_data

class TimestampMixinSchema(ChatSchema):
    createdAt: datetime = Field(description='Created At timestamp', default_factory=datetime.now)
    updatedAt: datetime = Field(description='Updated At timestamp', default_factory=datetime.now)