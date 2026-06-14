FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN adduser --disabled-password --gecos '' appuser && chown -R appuser /app
USER appuser

EXPOSE 5001

CMD ["gunicorn", "wsgi:app", \
     "--bind", "0.0.0.0:5001", \
     "--workers", "2", \
     "--timeout", "180", \
     "--access-logfile", "-"]
