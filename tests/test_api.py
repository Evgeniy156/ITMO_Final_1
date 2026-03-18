from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)


def test_healthcheck():
    response = client.get("/healthcheck")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_model_info():
    response = client.get("/model-info")
    # Ожидаем 200 OK, так как в CI модель должна быть загружена через dvc pull
    assert response.status_code == 200
    json_data = response.json()
    assert "model_type" in json_data
    assert "params" in json_data


def test_predict():
    # Валидные данные
    payload = {
        "fixed_acidity": 7.4,
        "volatile_acidity": 0.70,
        "citric_acid": 0.00,
        "residual_sugar": 1.9,
        "chlorides": 0.076,
        "free_sulfur_dioxide": 11.0,
        "total_sulfur_dioxide": 34.0,
        "density": 0.9978,
        "pH": 3.51,
        "sulphates": 0.56,
        "alcohol": 9.4
    }

    response = client.post("/predict", json=payload)

    # Ожидаем 200 OK, так как в CI модель должна быть загружена через dvc pull
    assert response.status_code == 200
    json_data = response.json()
    assert "prediction" in json_data
    assert isinstance(json_data["prediction"], float)


def test_predict_validation_error():
    # Не хватает обязательных полей
    payload = {
        "fixed_acidity": 7.4
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 422  # Pydantic validation error
