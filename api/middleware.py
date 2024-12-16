import os
import json
from datetime import datetime, timedelta
import jwt
from jwt import DecodeError
from contextvars import ContextVar
import requests
from requests.auth import HTTPBasicAuth
from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from .models import AsymmetricIdp

user_uuid_var = ContextVar('user_uuid')

def fetch_secret_key() -> str:
    idp = AsymmetricIdp(**json.loads(os.getenv('IDP')))
    return idp.get_key()

class MultiAuthorizationMiddleware(BaseHTTPMiddleware):
    """Middleware to capture only the first valid Bearer token Authorization header, removing 'undefined' headers."""
    async def dispatch(self, request: Request, call_next):
        import re
        bearer_pattern = re.compile(r"^Bearer\s+.+")
        headers = dict(request.scope['headers'])
        valid_auth_header = next(
            (auth for auth in [val.decode('utf-8') for key, val in headers.items() if key == b'authorization']
            if bearer_pattern.match(auth) and auth.lower() != "undefined"),
            None,
        )

        if valid_auth_header:
            authorization = valid_auth_header.encode()

            try:
                token = authorization.decode('utf-8').split('Bearer ')[1]
                decoded_jwt = jwt.decode(token, fetch_secret_key(), options={'verify_signature': True})
                user_uuid = decoded_jwt.get('sub', 'N/A')
                user_uuid_var.set(user_uuid)
            except DecodeError:
                raise ValueError(f'Invalid authorization header {authorization}')

            headers[b'authorization'] = authorization
            request.scope['headers'] = [(k, v) for k, v in headers.items()]
        else:
            raise HTTPException(status_code=401, detail='Missing or invalid Authorization header')

        response = await call_next(request)
        return response
    
class AddAuthorizationHeaderMiddleware(BaseHTTPMiddleware):
    """Edge middleware for local dev in certain circumstances"""
    async def dispatch(self, request: Request, call_next):
        authorization = request.headers.get('Authorization')
        new_token = None
        if authorization:
            token = authorization.replace('Bearer', '')
            if self.is_token_expired(token):
                new_token = self.refresh_app_state_token(request=request)
        else:
            new_token = self.refresh_app_state_token(request=request)
        headers = request.scope['headers']
        found = False
        for i, (k, v) in enumerate(headers):
            if k.lower() == b'authorization':
                headers[i] = (b'authorization', f'Bearer {new_token}'.encode())
                found = True
                break
        if not found:
            headers.append((b'authorization', f'Bearer {new_token}'.encode()))
        
        response = await call_next(request)
        return response

    def refresh_app_state_token(self, request: Request) -> str:
        """
        Returns a new token if the current app.state.token is expired, otherwise returns the current app.state.token.
        """
        if hasattr(request.app.state, 'token') == False or self.is_token_expired(request.app.state.token):
            new_token = self.get_new_token()
            request.app.state.token = new_token
            return new_token
        
        return request.app.state.token
    
    def is_token_expired(self, token: str) -> bool:
        try:
            payload = jwt.decode(token, options={'verify_signature': False})
            exp = datetime.fromtimestamp(payload.get('exp'))
            if exp < datetime.now() - timedelta(minutes=1):
                return True
            else:
                return False
        except:
            return True
        
    @staticmethod
    def get_new_token() -> str:
        username = os.getenv('JWT_ID')
        password = os.getenv('JWT_PASS')
        jwt_auth_url = os.getenv('JWT_AUTH_URL')
        s = requests.Session()
        s.verify = False
        url = jwt_auth_url
        auth = HTTPBasicAuth(username, password)
        response = s.post(url, auth=auth)
        if response.status_code == 200:
            token = response.json()['jwt']
            return token
        else:
            raise SystemError('Failed to get JWT')