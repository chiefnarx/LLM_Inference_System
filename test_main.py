from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_valid_completion_gpt2():
    response = client.post(
        "/v1/completions",
        json={
            "model": "gpt2",
            "prompt": "What is the capital of Morocco?",
            "max_tokens": 10
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert "choices" in data
    assert "text" in data["choices"][0]

def test_invalid_model():
    response = client.post(
        "/v1/completions",
        json={
            "model": "mistral", 
            "prompt": "What is the capital of Morocco?",
            "max_tokens": 10
        }
    )
    assert response.status_code == 404
    assert "Model" in response.json()["detail"]

def test_empty_prompt():
    response = client.post(
        "/v1/completions",
        json={
            "model": "gpt2",
            "prompt": " ",
            "max_tokens": 10
        }
    )
    assert response.status_code == 400
    assert "Prompt cannot be empty" in response.json()["detail"]
