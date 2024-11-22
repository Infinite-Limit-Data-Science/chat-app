import os
import logging
from ..langchain_logging import LangchainLogger
from .task_execution_context import filename_var

class FilenameFilter(logging.Filter):
    def filter(self, record):
        record.filename = filename_var.get()
        return True

_params = {}
if log_level := os.getenv('LOG_LEVEL'):
    _params['log_level'] = log_level
_params['log_format'] = '%(asctime)s {%(pathname)s:%(lineno)d} %(levelname)s - [File: %(filename)s] - %(message)s'
_params['log_name'] = 'langchain-doc'

logger = LangchainLogger(**_params)
core_logger = logger.load_config()

for handler in core_logger.handlers:
    if not any(isinstance(f, FilenameFilter) for f in handler.filters):
        handler.addFilter(FilenameFilter())