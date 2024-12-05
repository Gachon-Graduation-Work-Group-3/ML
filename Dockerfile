FROM python:3.11
WORKDIR /app

COPY requirements.txt /app/requirements.txt
COPY model/XGBoost_model.pkl /app/model/XGBoost_model.pkl
COPY service_api/main.py /app/main.py

RUN pip install --no-cache-dir -r /app/requirements.txt

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]