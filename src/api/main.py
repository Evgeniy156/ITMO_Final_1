from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import os
import logging
from sklearn.base import BaseEstimator

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Wine Quality Prediction API")

MODEL_PATH = "models/wine_quality_model.pkl"
model: BaseEstimator = None

class WineFeatures(BaseModel):
    fixed_acidity: float
    volatile_acidity: float
    citric_acid: float
    residual_sugar: float
    chlorides: float
    free_sulfur_dioxide: float
    total_sulfur_dioxide: float
    density: float
    pH: float
    sulphates: float
    alcohol: float

    # Пример валидации (можно добавить больше)
    model_config = {
        "json_schema_extra": {
            "example": {
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
        }
    }

@app.on_event("startup")
def load_model():
    global model
    # Попытка подтянуть модель через DVC (в реальном приложении лучше сделать это асинхронно или до старта сервиса)
    if not os.path.exists(MODEL_PATH):
        logger.info("Модель не найдена локально. Пытаемся загрузить через DVC...")
        os.system("dvc pull models/wine_quality_model.pkl.dvc")
    
    if os.path.exists(MODEL_PATH):
        try:
            with open(MODEL_PATH, "rb") as f:
                model = pickle.load(f)
            logger.info("Модель успешно загружена.")
        except Exception as e:
            logger.error(f"Ошибка при загрузке модели: {e}")
            raise RuntimeError(f"Не удалось загрузить модель: {e}")
    else:
        logger.warning("Модель не найдена даже после попытки dvc pull. Предсказания будут недоступны.")


@app.get("/health")
def healthcheck():
    return {"status": "ok"}

@app.get("/model-info")
def model_info():
    if model is None:
        raise HTTPException(status_code=503, detail="Модель не загружена")
    
    # В зависимости от того, какая модель загрузилась (LinearRegression/RandomForest)
    model_type = type(model).__name__
    info = {"model_type": model_type}
    
    if hasattr(model, "get_params"):
        info["params"] = model.get_params()
        
    return info

@app.post("/predict")
def predict(features: WineFeatures):
    if model is None:
        raise HTTPException(status_code=503, detail="Модель не загружена")
    
    # Конвертируем входные данные в формат, ожидаемый sklearn
    # Важно соблюдать порядок фичей, как в тренировочном датасете
    feature_dict = features.model_dump()
    
    # Pandas DataFrame нужен, чтобы передать те же названия колонок, что были при обучении
    import pandas as pd
    input_df = pd.DataFrame([feature_dict])
    
    try:
        prediction = model.predict(input_df)
        return {"prediction": float(prediction[0])}
    except Exception as e:
        logger.error(f"Ошибка при предсказании: {e}")
        raise HTTPException(status_code=400, detail=str(e))
