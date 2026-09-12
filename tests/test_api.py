from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_health():
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "ok"


def test_ask_requires_image():
    response = client.post(
        "/ask",
        data={"question": "What is the capital of France?"}
    )

    assert response.status_code == 422


def test_ask_requires_image_for_empty_question():
    response = client.post(
        "/ask",
        data={"question": ""}
    )

    assert response.status_code == 422