# Runs the Streamlit app anywhere that runs containers (Render, Railway, Google Cloud Run, Azure, a VPS...).
#   docker build -t bankruptcy-app .
#   docker run -p 8501:8501 bankruptcy-app        -> http://localhost:8501
FROM python:3.12-slim

# LightGBM needs the OpenMP runtime
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY .streamlit/ .streamlit/
COPY bankruptcy/ bankruptcy/
COPY models/ models/
COPY reports/figures/ reports/figures/
COPY app.py .

EXPOSE 8501
# Most hosts pass the port to listen on in $PORT
CMD ["sh", "-c", "streamlit run app.py --server.port=${PORT:-8501} --server.address=0.0.0.0"]
