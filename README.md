# Финальный проект по модулю MLOps

## Структура проекта

```text
/data          # Данные (wine_quality.csv), версионируются в DVC, не в Git
/models        # Обученные модели (.pkl), версионируются в DVC, не в Git
/dags          # Airflow DAGs (скрипт пайплайна wine_pipeline_dag.py)
/src
  /api         # FastAPI приложение (main.py - сервинг модели)
/tests         # Unit-тесты для API
.gitlab-ci.yml # Конфигурация GitLab CI/CD пайплайна
train.py       # Скрипт для экспериментов и обучения модели (настроен MLflow)
plan.md        # Чек-лист с пошаговым планом проекта
README.md      # Эта документация
requirements.txt # Зависимости Python
```

## Процесс и результаты экспериментов с моделями

Для предсказания качества вина были проведены эксперименты с двумя моделями машинного обучения (Задача регрессии). 
Трекинг экспериментов производился с помощью **MLflow**.

| Модель | Гиперпараметры | RMSE | MAE | R² |
| :--- | :--- | :--- | :--- | :--- |
| **Linear Regression** | Default | 0.6245 | 0.5035 | 0.4032 |
| **Random Forest Regressor** | n_estimators=100, max_depth=10, random_state=42 | **0.5641** | **0.4447** | **0.5131** |

**Обоснование выбора лучшей модели:**
Модель `Random Forest Regressor` показала значительно лучшие результаты на тестовой выборке по всем измеряемым метрикам: у нее ниже ошибка (RMSE и MAE) и выше коэффициент детерминации ($R^2$). Архитектура случайного леса позволяет лучше улавливать нелинейные зависимости в характеристиках вин (таких как кислотность, уровень сахара и алкоголь), поэтому именно эта модель была сохранена (`models/wine_quality_model.pkl`) и используется в FastAPI сервисе (для предсказаний) и закреплена в DVC.

## Инструкции по локальному запуску

### 1. Подготовка окружения
Клонируйте репозиторий и установите все зависимости:
```bash
python -m venv venv
venv\Scripts\activate # (на Windows)
pip install -r requirements.txt
```

### 2. Настройка DVC и MinIO
Убедитесь, что у вас запущен S3 совместимый сервер MinIO (например через Docker):
```bash
dvc remote add -d minio s3://wine-quality
dvc remote modify minio endpointurl http://127.0.0.1:9000
```
В рабочей среде также задайте данные вашей учётной записи (установите ключи AWS!):
```bash
# В Windows PowerShell:
$env:AWS_ACCESS_KEY_ID="minioadmin"
$env:AWS_SECRET_ACCESS_KEY="minioadmin"
```
Подтяните исходные данные и обученную модель:
```bash
dvc pull
```

### 3. Запуск Airflow DAG (Опционально)
Для автоматизации пайплайна (`dags/wine_pipeline_dag.py`) скопируйте папку `dags/` в директорию вашего инстанса Airflow. DAG автоматически:
1. Подтягивает актуальные данные из DVC (`dvc pull`).
2. Обучает модель (запуская скрипт `train.py`).
3. Сохраняет модель и коммитит метаданные DVC в Git.

### 4. Запуск FastAPI сервера 
Для запуска API для сервинга:
```bash
uvicorn src.api.main:app --reload
```
Доступные эндпоинты по умолчанию (`http://127.0.0.1:8000`):
- `/health` — статус сервера
- `/model-info` — сведения об актуальной версии модели
- `/predict` — получение предсказаний по POST параметрам (в JSON теле)

### 5. Запуск тестов
Для проверки работоспособности API запустите тесты при помощи `pytest`:
```bash
set PYTHONPATH=.
pytest tests/
```
*(Tests mock and test requests without real models directly, if proper structure implemented)*
