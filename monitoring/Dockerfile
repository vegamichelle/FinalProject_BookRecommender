FROM python:3.12-slim
WORKDIR /app
COPY monitoring/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY src ./src
COPY monitoring ./monitoring
ENV PYTHONPATH=/app
EXPOSE 8502
CMD ["streamlit", "run", "monitoring/dashboard.py", "--server.port=8502", "--server.address=0.0.0.0"]
