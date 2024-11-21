import os
import logging
from ..langchain_logging import LangchainLogger
from .task_execution_context import session_id_var

class SessionIdFilter(logging.Filter):
    def filter(self, record):
        record.session_id = session_id_var.get()
        return True

_params = {}
if log_level := os.getenv('LOG_LEVEL'):
    _params['log_level'] = log_level
_params['log_format'] = '%(asctime)s {%(pathname)s:%(lineno)d} %(levelname)s - [Session Id: %(session_id)s] - %(message)s'
_params['log_name'] = 'langchain-chat'

logger = LangchainLogger(**_params)
core_logger = logger.load_config()

for handler in core_logger.handlers:
    if not any(isinstance(f, SessionIdFilter) for f in handler.filters):
        handler.addFilter(SessionIdFilter())