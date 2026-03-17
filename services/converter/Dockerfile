FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY services /app/services

ENV PORT=8080
CMD ["python","-m","uvicorn","services.converter.app:app","--host","0.0.0.0","--port","8080"]

