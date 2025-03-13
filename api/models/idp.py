import os
import json
import requests
from pathlib import Path
from urllib.parse import urlparse
from typing import Literal, Optional, Callable
from jwt.algorithms import RSAAlgorithm
from abc import ABC, abstractmethod
from pydantic import Field, HttpUrl, field_validator
from .mongo_schema import ChatSchema


def fetch_public_key() -> str:
    idp = AsymmetricIdp(**json.loads(os.getenv("IDP")))
    return idp.get_key()


class Idp(ChatSchema, ABC):
    source: Literal["microsoft_entra", "forgerock"]
    algorithm: Literal["HS256", "RS256"]

    @abstractmethod
    def get_key(self) -> str | Path: ...


class SymmetricIdp(Idp):
    algorithm: Literal["HS256"]
    secret: str = Field(
        description="Shared secret for symmetric algorithm", min_length=8
    )

    @field_validator("secret", mode="before")
    @classmethod
    def validate_secret(cls, value):
        if not value.strip():
            raise ValueError("The `secret` cannot be empty.")
        if not value.isascii():
            raise ValueError("The `secret` must contain only ASCII characters.")
        return value

    def get_key(self) -> str:
        return self.secret


class AsymmetricIdp(Idp):
    source: str
    algorithm: Literal["RS256"]
    key_source: Literal["file", "url"] = Field(
        ..., description="The source of the public key"
    )
    public_key: Optional[Path] = Field(
        description="Path to the public key file if key_source is file", default=None
    )
    public_key_url: Optional[HttpUrl] = Field(
        description="URL to fetch JWKS if key_source is url", default=None
    )

    def __init__(self, **data):
        super().__init__(**data)
        self._key_handlers: dict[str, Callable[[], str]] = {
            "file": self._load_key_from_file,
            "url": self._load_key_from_url,
        }

    @field_validator("public_key", mode="before")
    @classmethod
    def convert_to_path(cls, value):
        if value is not None and isinstance(value, str):
            return Path(value)
        raise ValueError("Expected Path-like object")

    @field_validator("public_key_url", mode="before")
    @classmethod
    def validate_url(cls, value):
        parsed = urlparse(value)
        if not all([parsed.scheme, parsed.netloc]):
            raise ValueError("Invalid URL format")
        return value

    def _load_key_from_file(self) -> str:
        if not self.public_key or not self.public_key.exists():
            raise FileNotFoundError(f"Public key file not found: {self.public_key}")

        return self.public_key.read_bytes()

    def _load_key_from_url(self) -> str:
        jwks = requests.get(str(self.public_key_url)).json()
        return self._extract_public_key_from_jwks(jwks)

    def get_key(self) -> str:
        handler = self._key_handlers.get(self.key_source)
        if not handler:
            raise ValueError(f"Unsupported key_source: {self.key_source}")
        return handler()

    @staticmethod
    def _extract_public_key_from_jwks(jwks: dict) -> str:
        key = jwks["keys"][0]
        return RSAAlgorithm.from_jwk(json.dumps(key))
