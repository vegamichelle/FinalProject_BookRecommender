FROM python:3.12-slim
WORKDIR /app
COPY backend/requirements.txt ./backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt
COPY src ./src
COPY backend ./backend
ENV PYTHONPATH=/app
EXPOSE 8000
# Model loaded from the W&B registry (MODEL_SOURCE=wandb) or a mounted
# artifacts/ volume (MODEL_SOURCE=local). See README.
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
