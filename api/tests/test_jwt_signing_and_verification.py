import jwt
import pytest
import os
from collections import namedtuple
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives import serialization
from tempfile import NamedTemporaryFile

JWTPayload = namedtuple("JWTPayload", [
    "app", "sub", "mail", "src", "roles", "iss", "attributes", "aud", 
    "givenname", "displayname", "sn", "idm_picture_url", "exp", "iat", 
    "session_id", "jti"
])

@pytest.fixture
def rsa_key_pair():
    private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048
    )
    
    public_key = private_key.public_key()

    private_key_file = NamedTemporaryFile(delete=False, suffix=".pem")
    public_key_file = NamedTemporaryFile(delete=False, suffix=".pem")

    try:
        private_pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        )
        private_key_file.write(private_pem)
        private_key_file.close()

        public_pem = public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
        public_key_file.write(public_pem)
        public_key_file.close()

        yield private_key_file.name, public_key_file.name
    finally:
        os.unlink(private_key_file.name)
        os.unlink(public_key_file.name)

@pytest.fixture
def jwt_payload():
    payload = JWTPayload(
        app="svc-chat-test",
        sub="n1m4",
        mail="john.doe@bcbsfl.com",
        src="john.doe@bcbsfl.com",
        roles=[""],
        iss="PMI-Test",
        attributes=[
            {"givenname": "John"},
            {"sn": "JDoe"},
            {"mail": "john.doe@bcbsfl.com"},
            {"displayname": "JDoe, John"},
            {"bcbsfl-idmPictureURL": ""}
        ],
        aud="chatapp-tsta.throtl.com",
        givenname="John",
        displayname="Doe, John",
        sn="JDoe",
        idm_picture_url="",
        exp=1893456000,
        iat=1714144841,
        session_id="",
        jti=""
    )

    return payload._asdict()

def test_jwt_signing_and_verification(rsa_key_pair, jwt_payload):
    """
    In JWT signing with asymmetric cryptography, such as RS256, 
    you use the private key to sign the JWT and the public key to 
    verify it.
    """
    private_key_path, public_key_path = rsa_key_pair

    with open(private_key_path, "rb") as f:
        private_key = f.read()

    token = jwt.encode(jwt_payload, private_key, algorithm="RS256")

    with open(public_key_path, "rb") as f:
        public_key = f.read()

    decoded_payload = jwt.decode(token, public_key, algorithms=["RS256"], audience=jwt_payload["aud"])

    assert decoded_payload == jwt_payload