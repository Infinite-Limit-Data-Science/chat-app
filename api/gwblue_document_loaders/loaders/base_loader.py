import os
from typing import Union, Iterator
from pathlib import Path
import tempfile
import requests
from urllib.parse import urlparse
from langchain_core.documents import Document
from abc import ABC, abstractmethod


class BaseLoader(ABC):
    def __init__(self, file_path: Union[str, Path]):
        self._file_path = str(file_path)
        if "~" in self._file_path:
            self.file_path = os.path.expanduser(self._file_path)

        if not os.path.isfile(self._file_path) and self._is_valid_url(self._file_path):
            res = requests.get(self._file_path)

            if res.status_code != 200:
                raise ValueError(
                    "Invalid URL; returned status code %s" % res.status_code
                )

            self._web_path = self._file_path
            self._temp_file = tempfile.NamedTemporaryFile()
            self._temp_file.write(res.content)
            self._file_path = self._temp_file.name
        elif not os.path.isfile(self._file_path):
            raise ValueError(
                "File path %s is not a valid file or url" % self._file_path
            )

    def __del__(self) -> None:
        if hasattr(self, "_temp_file"):
            self._temp_file.close()

    @abstractmethod
    def load(self) -> Iterator[Document]:
        pass

    @staticmethod
    def _is_valid_url(url: str) -> bool:
        """Check if the url is valid."""
        parsed = urlparse(url)
        return bool(parsed.netloc) and bool(parsed.scheme)

    def sourcify(self, path: Union[str, Path]) -> str:
        return os.path.basename(str(path))
