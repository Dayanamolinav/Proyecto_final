# OncoRisk · imagen del prototipo (arquitectura física VII-B.8)
# Construir:  docker build -t oncorisk .
# Ejecutar:   docker run --rm -p 8501:8501 oncorisk      ->  http://localhost:8501
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DATA_BACKEND=local

# libgomp1: requerido por XGBoost en imágenes slim
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

# Se reentrena dentro de la imagen con el lote sintético local: así el modelo
# guardado coincide con las versiones de scikit-learn/XGBoost instaladas aquí.
RUN python scripts/run_pipeline.py --backend local --usar-csv-local \
    && python -m unittest discover -s tests

RUN useradd --create-home --uid 1000 oncorisk && chown -R oncorisk /app
USER oncorisk

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

CMD ["python", "-m", "streamlit", "run", "app/streamlit_app.py", \
     "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
