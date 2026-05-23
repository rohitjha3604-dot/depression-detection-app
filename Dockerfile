FROM python:3.9-slim

# Install system libraries (opensmile needs libsndfile, xgboost/lightgbm need libgomp)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir torch==2.0.1 --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Pre-download NLTK data so the app doesn't crash on first tokenization
RUN python -m nltk.downloader punkt stopwords wordnet averaged_perceptron_tagger

# Copy the entire app including models
COPY . .

EXPOSE 8501

CMD streamlit run app.py --server.port=$PORT --server.address=0.0.0.0 --server.headless=true
