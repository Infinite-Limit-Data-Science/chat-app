import os
from pathlib import Path
from fastapi.testclient import TestClient
from ..main import app


def test_create_conversation_with_pdf():
    with TestClient(app) as client:
        file_path = Path(__file__).parent / "assets" / "NVIDIAAn.pdf"

        data = {"content": "Summarize the document."}

        files = {
            "upload_files": ("NVIDIAAn.pdf", open(file_path, "rb"), "application/pdf")
        }

        token = os.getenv("TEST_AUTH_TOKEN")
        headers = {"Authorization": f"Bearer {token}"}

        response = client.post(
            "/api/conversations/", data=data, files=files, headers=headers
        )
        print(response.text)
        assert response.status_code == 200


def test_create_conversation_with_image():
    with TestClient(app) as client:
        file_path = Path(__file__).parent / "assets" / "baby.jpg"
        file_path2 = Path(__file__).parent / "assets" / "guitar.jpg"

        data = {"content": "Compare and contrast the images."}

        files = [
            (
                "upload_files",
                ("baby.jpg", open(file_path, "rb"), "image/jpeg"),
            ),
            (
                "upload_files",
                ("persob.jpg", open(file_path2, "rb"), "image/jpeg"),
            ),
        ]

        token = os.getenv("TEST_AUTH_TOKEN")
        headers = {"Authorization": f"Bearer {token}"}

        response = client.post(
            "/api/conversations/", data=data, files=files, headers=headers
        )

        print(response.text)
        assert response.status_code == 200
