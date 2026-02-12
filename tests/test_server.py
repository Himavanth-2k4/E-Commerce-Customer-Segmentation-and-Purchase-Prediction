import pytest
from fastapi.testclient import TestClient
from server import app

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "model_loaded" in data

def test_static_files():
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers.get("content-type", "")

def test_predict_validation_error():
    # Sending invalid data (empty body) should return 422
    response = client.post("/predict", json={})
    assert response.status_code == 422
    data = response.json()
    assert data["status"] == "error"
    assert "Validation error" in data["error"]

def test_predict_valid_payload():
    payload = {
        "Gender": "Female",
        "Age": 30,
        "City": "New York",
        "Membership_Type": "Gold",
        "Total_Spend": 500.0,
        "Items_Purchased": 5,
        "Average_Rating": 4.0,
        "Days_Since_Last_Purchase": 10
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "prediction" in data
    assert "satisfaction_level" in data


def test_predict_batch():
    batch_payload = [
        {
            "Gender": "Female",
            "Age": 28,
            "City": "Los Angeles",
            "Membership_Type": "Silver",
            "Total_Spend": 350.0,
            "Items_Purchased": 3,
            "Average_Rating": 3.5,
            "Days_Since_Last_Purchase": 15
        },
        {
            "Gender": "Male",
            "Age": 45,
            "City": "Chicago",
            "Membership_Type": "Gold",
            "Total_Spend": 1200.0,
            "Items_Purchased": 12,
            "Average_Rating": 4.8,
            "Days_Since_Last_Purchase": 4
        }
    ]
    response = client.post("/predict/batch", json=batch_payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["count"] == 2
    assert len(data["results"]) == 2
