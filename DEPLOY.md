# 🚀 Deployment Guide — Streamlit Community Cloud

This app can be hosted for **free** on [Streamlit Community Cloud](https://streamlit.io/cloud).

---

## Option 1: Streamlit Community Cloud (Recommended — Free)

### Step 1: Push to GitHub

1. Create a new public repository on GitHub (e.g., `multimodal-depression-detection`)
2. Upload this project:

```bash
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/multimodal-depression-detection.git
git push -u origin main
```

> **Note on file sizes:** The `saved_models/` folder is ~119MB total. All individual files are under GitHub's 100MB limit, so no Git LFS is needed. The entire repo should be well under Streamlit Cloud's limits.

### Step 2: Deploy on Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with your GitHub account
3. Click **"New app"**
4. Select your repository, branch `main`, and main file path: `app.py`
5. Click **Deploy**

Streamlit Cloud will automatically:
- Install system packages from `packages.txt` (`libsndfile1` for audio support)
- Install Python packages from `requirements.txt`
- Download NLTK data on first run (handled in `app.py`)

### Step 3: Access Your App

Your app will be available at:
```
https://your-app-name.streamlit.app
```

---

## Option 2: Self-Hosted (Docker / VPS)

If you prefer your own server, use this Dockerfile:

```dockerfile
FROM python:3.9-slim

# Install system dependencies for audio processing
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

Build and run:
```bash
docker build -t depression-app .
docker run -p 8501:8501 depression-app
```

---

## ⚠️ Important Notes

| Topic | Details |
|-------|---------|
| **scikit-learn version** | Must stay at `1.4.2` (pinned in `requirements.txt`). The pickled models were saved with this version. Upgrading will break model loading. |
| **Audio analysis** | Requires OpenSMILE + `libsndfile1`. The `packages.txt` ensures this installs on Streamlit Cloud. |
| **NLTK data** | Downloaded automatically on first run. No manual setup needed. |
| **First startup** | May take 30–60 seconds due to large model files (PyTorch + sklearn ensembles). |
| **Memory** | Peak RAM usage is ~800MB–1GB. Streamlit Cloud free tier provides 1GB, which is sufficient. |

---

## 🔒 Privacy Disclaimer

The app includes a built-in disclaimer that it is for **research and educational purposes only** and should not replace professional medical diagnosis.

---

## 📊 Model Performance Summary

| Modality | Best Model | Accuracy | F1 Score |
|----------|-----------|----------|----------|
| Text (DL) | Attention BiLSTM | 72.92% | **74.05%** |
| Text (ML) | Stacking Ensemble | 73.76% | 73.33% |
| Audio | Improved 3-Layer CNN | **100.00%** (file-level) | **100.00%** |
