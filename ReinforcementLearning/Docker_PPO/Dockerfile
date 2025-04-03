FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY inference.py .

COPY ppo_microgrid_best.zip .

COPY data.zip .

CMD ["python", "inference.py"]
