# 🧠 MiSec — Multimodal Depression Detection

A Streamlit web app that detects signs of depression by analyzing **audio (speech)** and **text** inputs using deep learning models, with **Explainable AI (XAI)** and **LLM-powered clinical analysis**.

![Streamlit](https://img.shields.io/badge/Streamlit-1.50.0-FF4B4B?logo=streamlit)
![Python](https://img.shields.io/badge/Python-3.9-3776AB?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-EE4C2C?logo=pytorch)

---

## 🚀 Quick Start (Local)

### 1. Clone the repo

```bash
git clone https://github.com/rohitjha3604-dot/depression-detection-app.git
cd depression-detection-app
```

### 2. Create a virtual environment

```bash
python3.9 -m venv venv
source venv/bin/activate        # macOS/Linux
# OR
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** PyTorch is pinned to CPU-only version in `requirements.txt`. If you have a GPU, adjust the torch install command accordingly.

### 4. Set your OpenAI API key (optional)

The **AI-Powered Deep Analysis** feature uses GPT-4o-mini. If you skip this step, the app runs fine but the LLM analysis section won't appear.

```bash
export OPENAI_API_KEY="sk-..."
```

> On Windows: `set OPENAI_API_KEY=sk-...`

### 5. Run the app

```bash
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

---

## 🐳 Run with Docker

```bash
# Build
docker build -t misec-app .

# Run
docker run -p 8501:8501 -e OPENAI_API_KEY="sk-..." misec-app
```

Open **http://localhost:8501**.

---

## ☁️ Deploy to Render

1. Push this repo to GitHub
2. Go to [dashboard.render.com](https://dashboard.render.com) → **New +** → **Web Service**
3. Connect your GitHub repo
4. Select **Docker** as the runtime
5. Add environment variable: `OPENAI_API_KEY = sk-...`
6. Click **Create Web Service**

Render will build from the `Dockerfile` and deploy automatically.

---

## 📁 Project Structure

```
├── app.py                    # Main Streamlit app (landing page + dashboard)
├── saved_models/             # Trained model artifacts (~120MB)
│   ├── improved_audio_cnn.pth
│   ├── lstm_model.pth
│   ├── best_text_model.pkl
│   └── ...
├── audiomodels/              # Audio model architectures
├── decision/                 # Fusion strategies (Weighted, Stacking, etc.)
├── extract_audio.py          # Audio feature extraction (OpenSMILE)
├── extractBERT.py            # BERT feature extraction
├── train_text_model.py       # Text model training pipeline
├── requirements.txt          # Python dependencies
└── Dockerfile                # Render deployment
```

---

## 🎯 Features

| Feature | Description |
|---------|-------------|
| 🎤 **Audio Analysis** | Upload WAV or record live voice. CNN analyzes OpenSMILE eGeMAPS features. |
| 📝 **Text Analysis** | Enter text. Choose from Stacking Ensemble, Soft Voting, or Attention BiLSTM. |
| 🔀 **Multimodal Fusion** | Combines audio + text with configurable weights. |
| 🔍 **Explainable AI** | See which audio features or words drove the prediction. |
| 🤖 **AI Deep Analysis** | GPT-4o-mini generates a structured clinical report (requires API key). |

---

## ⚠️ Disclaimer

This tool is for **research and educational purposes only**. It is **not** a substitute for professional medical diagnosis or mental health counseling.

---

## 🛠️ Tech Stack

- **Frontend:** Streamlit
- **Deep Learning:** PyTorch
- **ML Models:** scikit-learn, XGBoost, LightGBM
- **Audio Features:** OpenSMILE (eGeMAPSv02)
- **NLP:** NLTK, TF-IDF
- **LLM:** OpenAI GPT-4o-mini
- **Visualization:** Plotly
