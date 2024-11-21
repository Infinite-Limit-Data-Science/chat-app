from typing import Optional, List, Self, TextIO, Any
import itertools as it
from collections import namedtuple
import re
import logging
from logging import Logger, Handler, StreamHandler, FileHandler
from pathlib import Path
from pydantic import (
    BaseModel, Field, field_validator, model_validator)

def gen_sequence(base: int) -> List[int]:
    sequence = it.count(start=base, step=10)
    return [
        num
        for _, num in zip(range(5), sequence)
    ]

_LOG_LEVELS = namedtuple('LOG_LEVEL', ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'])
_log_levels = _LOG_LEVELS(*gen_sequence(10))

class LangchainLogger(BaseModel):
    """Specialized logging class; in the immenent future, it will also comprise custom logging for Runnables"""
    log_level: str = Field(description='Specify Log Level, one of DEBUG, INFO, WARNING, ERROR, CRITICAL', default='INFO')
    log_file: Optional[str] = Field(description='Specify Log File', default=None)
    log_format: Optional[str] = Field(description='Format for logging', default='%(asctime)s {%(pathname)s:%(lineno)d} %(levelname)s - [User: %(user_uuid)s] - %(message)s')
    log_name: Optional[str] = Field(description='Log name of specific log instance', default=None)
    base_logger: Logger = Field(description='Underlying logger to proxy to', default=None)
    load_params: dict = Field(description='Logger params', default_factory=dict)

    class Config:
        arbitrary_types_allowed = True        
    
    @field_validator('log_level', mode='before')
    @classmethod
    def normalize_log_level(cls, value: str) -> str:
        """Validate Log Level"""
        normalized_value = value.upper()
        if normalized_value not in _log_levels._fields:
            raise ValueError(f'Invalid log level: {value}. Must be one of {', '.join(_log_levels._fields)}.')
        return normalized_value
    
    @field_validator('log_level', mode='before')
    @classmethod
    def normalize_log_file(cls, value: str | None) -> str:
        """Allow for a log_file or log_file.log where log_file is name of log file"""
        if value:
            log_file = Path(value)
            if log_file.suffix and log_file.suffix != '.log':
                value = log_file.stem + '.log'

        return value
    
    @field_validator('log_format', mode='before')
    @classmethod
    def validate_log_format(cls, value: str) -> str:
        pattern = r"%\((asctime|levelname|message)\)\w*"
        match = re.search(pattern, value)
        if not match:
            raise ValueError(f'Invalid log format pattern {value}')

        return value            

    @model_validator(mode='after')
    def validate_logger_for_load(self) -> Self:
        self.load_params = {}
        
        if self.log_file:
            self.load_params['filename'] = self.log_file
        self.load_params['log_level'] = _log_levels._asdict()[self.log_level]
        self.load_params['log_format'] = self.log_format

        return self
    
    def set_log_stream(self) -> StreamHandler[TextIO]:
        formatter = logging.Formatter(self.log_format)
        handler = logging.StreamHandler()
        handler.setLevel(self.load_params['log_level'])
        handler.setFormatter(formatter)

        return handler

    def set_log_file(self) -> FileHandler:
        formatter = logging.Formatter(self.log_format)
        handler = logging.FileHandler(self.log_file)
        handler.setLevel(self.load_params['log_level'])
        handler.setFormatter(formatter)

        return handler

    def load_config(self) -> Logger:
        self.base_logger = logging.getLogger(self.log_name or __name__)
        self.base_logger.setLevel(self.load_params['log_level'])

        self.base_logger.propagate = False
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)

        handler: Handler = self.set_log_file() if self.log_file else self.set_log_stream()
        self.base_logger.addHandler(handler)

        return self.base_logger

    def load_basic(self) -> None:
        """Load basic config; not recommended"""
        logging.basicConfig(**self.load_params)

    def __getattr__(self, name: str) -> Any:
        """method_missing"""
        if self.base_logger and hasattr(self.base_logger, name):
            return getattr(self.base_logger, name)
        
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")