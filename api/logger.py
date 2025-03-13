import os
import logging
from .langchain_logging import LangchainLogger
import warnings
from contextvars import ContextVar

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="pydantic",
    message=r"^Pydantic serializer warnings:\s+Expected `str` but got `ObjectId`",
)

user_uuid_var = ContextVar("user_uuid")


class UserUUIDFilter(logging.Filter):
    def filter(self, record):
        record.user_uuid = user_uuid_var.get()
        return True


_params = {}
if log_level := os.getenv("LOG_LEVEL"):
    _params["log_level"] = log_level
if log_file := os.getenv("LOG_FILE"):
    _params["log_file"] = log_file
if log_format := os.getenv("LOG_FORMAT"):
    _params["log_format"] = log_format
_params["log_name"] = "chat-api"

logger = LangchainLogger(**_params)
core_logger = logger.load_config()

for handler in core_logger.handlers:
    if not any(isinstance(f, UserUUIDFilter) for f in handler.filters):
        handler.addFilter(UserUUIDFilter())
