FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1     PYTHONDONTWRITEBYTECODE=1

RUN pip install --no-cache-dir torch==2.0.1 --index-url https://download.pytorch.org/whl/cpu  --extra-index-url https://pypi.org/simple && \
    pip install --no-cache-dir pandas requests "numpy<2"
    
WORKDIR /app
COPY model.py client.py server.py /app/
