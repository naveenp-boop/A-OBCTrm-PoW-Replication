FROM python:3.14-slim

WORKDIR /app
COPY requirements-lock.txt .
RUN pip install --no-cache-dir -r requirements-lock.txt

COPY sim/ ./sim/
COPY scripts/ ./scripts/
COPY tests/ ./tests/

CMD ["python", "scripts/run_all_experiments.py"]
