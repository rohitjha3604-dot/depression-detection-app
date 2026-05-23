# 🧠 Multimodal Depression Detection

A web application that detects signs of depression by analyzing both **audio (speech)** and **text** inputs using deep learning and machine learning models, with built-in **Explainable AI (XAI)**.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-name.streamlit.app)

---

## 🎯 Features

- 🎤 **Audio Analysis** — Upload a WAV file. An improved 3-layer CNN with BatchNorm analyzes OpenSMILE eGeMAPSv02 speech features (pitch, jitter, shimmer, MFCCs, loudness).
- 📝 **Text Analysis** — Enter any text. Three models available:
  - **Best ML Model** — Stacking Ensemble (XGBoost + LightGBM + Gradient Boosting + SVM → Logistic Regression meta)
  - **Ensemble** — Soft Voting Ensemble
  - **Attention BiLSTM** — Deep learning with self-attention mechanism
- 🔀 **Multimodal Fusion** — Combines audio + text predictions with configurable weights.
- 🔍 **Explainable AI** — See which audio features or words drove the prediction.

---

## 📊 Model Performance

| Modality | Best Model | Accuracy | F1 Score |
|----------|-----------|----------|----------|
| **Text (DL)** | Attention BiLSTM | 72.92% | **74.05%** |
| **Text (ML)** | Stacking Ensemble | 73.76% | 73.33% |
| **Audio** | Improved 3-Layer CNN | **100.00%** (file-level) | **100.00%** |

---

## 🚀 Quick Start (Local)

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/multimodal-depression-detection.git
cd multimodal-depression-detection

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

Then open `http://localhost:8501` in your browser.

> **macOS users:** If XGBoost/LightGBM fail to load, install OpenMP:
> ```bash
> brew install libomp
> ```

---

## ☁️ Deploy to Streamlit Community Cloud

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Set main file to `app.py`
5. Deploy

See [`DEPLOY.md`](DEPLOY.md) for detailed instructions and Docker setup.

---

## 📁 Project Structure

```
├── app.py                    # Main Streamlit application
├── saved_models/             # Trained model artifacts (~119MB)
│   ├── improved_audio_cnn.pth
│   ├── lstm_model.pth
│   ├── best_text_model.pkl
│   ├── tfidf_vectorizer.pkl
│   └── ...
├── audiomodels/              # Audio model architectures (CNN, LSTM, BiLSTM, etc.)
├── decision/                 # Multimodal fusion strategies
├── feturelevelfusion/        # Feature-level fusion models
├── train_text_model.py       # Text model training pipeline
├── extract_audio*.py         # Audio feature extraction scripts
├── requirements.txt          # Python dependencies
├── packages.txt              # System dependencies (for Streamlit Cloud)
└── DEPLOY.md                 # Deployment guide
```

---

## ⚠️ Disclaimer

This tool is for **research and educational purposes only**. It should **not** be used as a substitute for professional medical diagnosis or mental health counseling.

---

## 🛠️ Tech Stack

- **Frontend:** Streamlit
- **Deep Learning:** PyTorch
- **ML Models:** scikit-learn, XGBoost, LightGBM
- **Audio Features:** OpenSMILE (eGeMAPSv02)
- **NLP:** NLTK, TF-IDF
- **Visualization:** Plotly

---

## 📄 License

MIT License — feel free to use and modify for research purposes.
# depression-detection-app
