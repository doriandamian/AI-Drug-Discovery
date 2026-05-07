FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    build-essential \
    libglib2.0-0 \
    libxext6 \
    libsm6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# 4. Creăm un director de lucru în interiorul containerului
WORKDIR /app

# 5. Copiem întâi DOAR fișierul de cerințe (ajută la caching-ul de construire Docker)
COPY requirements.txt /app/

# 6. Instalăm pachetele Python
RUN pip install --no-cache-dir -r requirements.txt

# 7. Copiem restul codului sursă în container (app.py, /agents, /tools, etc.)
COPY . /app/

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]