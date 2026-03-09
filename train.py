import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error, r2_score, mean_absolute_error
import mlflow
import mlflow.sklearn

os.makedirs('models', exist_ok=True)

# Загрузка данных
print("Загрузка данных...")
df = pd.read_csv('data/wine_quality.csv')
X = df.drop(columns=['quality'])
y = df['quality']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def evaluate_metrics(y_true, y_pred):
    rmse = root_mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return rmse, mae, r2

# Настройка MLflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Wine Quality Prediction")

best_model = None
best_rmse = float('inf')
best_model_name = ""

# Эксперимент 1: Linear Regression
print("Запуск эксперимента: Linear Regression")
with mlflow.start_run(run_name="Linear Regression"):
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    
    rmse, mae, r2 = evaluate_metrics(y_test, y_pred)
    
    mlflow.log_param("model_type", "LinearRegression")
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("r2", r2)
    
    mlflow.sklearn.log_model(lr, "model")
    
    if rmse < best_rmse:
        best_rmse = rmse
        best_model = lr
        best_model_name = "LinearRegression"
        
    print(f"Linear Regression - RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")

# Эксперимент 2: Random Forest
print("Запуск эксперимента: Random Forest Regressor")
n_estimators = 100
max_depth = 10
with mlflow.start_run(run_name="Random Forest"):
    rf = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    
    rmse, mae, r2 = evaluate_metrics(y_test, y_pred)
    
    mlflow.log_param("model_type", "RandomForestRegressor")
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("r2", r2)
    
    mlflow.sklearn.log_model(rf, "model")
    
    if rmse < best_rmse:
        best_rmse = rmse
        best_model = rf
        best_model_name = "RandomForest"
        
    print(f"Random Forest - RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")

# Сохранение лучшей модели
print(f"Лучшая модель: {best_model_name} с RMSE = {best_rmse:.4f}")
model_path = 'models/wine_quality_model.pkl'
with open(model_path, 'wb') as f:
    pickle.dump(best_model, f)
print(f"Лучшая модель сохранена в {model_path}")
