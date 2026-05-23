"""
Multimodal Depression Detection Web App with XAI
Combines Audio (Improved CNN) + Text (NLP with Attention BiLSTM + Ensemble)
for comprehensive detection with Explainable AI features
"""

import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import opensmile
import os
import tempfile
import pickle
import re
from collections import Counter
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# NLP
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Download NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)

# Configure page
st.set_page_config(
    page_title="Multimodal Depression Detection",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .modality-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.5rem;
        border-radius: 1rem;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .audio-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 1rem;
    }
    .text-card {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 1rem;
    }
    .fusion-card {
        background: linear-gradient(135deg, #ff6b6b 0%, #feca57 100%);
        color: white;
        padding: 2rem;
        border-radius: 1.5rem;
        text-align: center;
    }
    .prediction-positive {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 2rem;
        border-radius: 1rem;
        color: white;
        text-align: center;
    }
    .prediction-negative {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 2rem;
        border-radius: 1rem;
        color: white;
        text-align: center;
    }
    .xai-section {
        background: white;
        padding: 1.5rem;
        border-radius: 1rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    .metric-improved {
        color: #38ef7d;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ============== IMPROVED AUDIO MODEL ==============
class ImprovedCNNModel(nn.Module):
    """Deeper CNN with BatchNorm, multiple conv layers, and global average pooling"""
    def __init__(self, num_channels=25):
        super(ImprovedCNNModel, self).__init__()
        self.conv1 = nn.Conv1d(num_channels, 64, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(256, 64)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool1(self.relu(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu(self.bn2(self.conv2(x))))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.global_pool(x)
        x = x.squeeze(-1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return self.sigmoid(x)

# Fallback simple model for backward compatibility
class AudioCNNModel(nn.Module):
    def __init__(self, num_channels=25):
        super(AudioCNNModel, self).__init__()
        self.layer1 = nn.Conv1d(num_channels, 64, kernel_size=5, stride=1, padding=2)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.drop_out = nn.Dropout(0.5)
        self.fc1 = nn.Linear(64 * 64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.pool(torch.relu(self.layer1(x)))
        x = x.view(x.size(0), -1)
        x = self.drop_out(x)
        x = self.fc1(x)
        return self.sigmoid(x)

# ============== IMPROVED TEXT MODEL (Attention BiLSTM) ==============
class Attention(nn.Module):
    """Self-attention layer for LSTM outputs"""
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )
    
    def forward(self, lstm_output):
        attention_weights = self.attn(lstm_output)
        attention_weights = torch.softmax(attention_weights, dim=1)
        context = torch.sum(attention_weights * lstm_output, dim=1)
        return context, attention_weights


class AttentionBiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, hidden_dim=256, num_layers=2):
        super(AttentionBiLSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim, num_layers, 
            batch_first=True, bidirectional=True, dropout=0.3
        )
        self.attention = Attention(hidden_dim * 2)
        self.fc1 = nn.Linear(hidden_dim * 2, 128)
        self.fc2 = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.4)
        self.ln = nn.LayerNorm(hidden_dim * 2)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        embedded = self.embedding(x)
        embedded = self.dropout(embedded)
        lstm_out, _ = self.lstm(embedded)
        lstm_out = self.ln(lstm_out)
        context, attn_weights = self.attention(lstm_out)
        out = self.dropout(context)
        out = self.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.fc2(out)
        return self.sigmoid(out).squeeze()

# Keep old LSTM for backward compatibility
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=128, num_layers=2):
        super(LSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, 
                           batch_first=True, bidirectional=True, dropout=0.3)
        self.fc = nn.Linear(hidden_dim * 2, 1)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, (hidden, _) = self.lstm(embedded)
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        hidden = self.dropout(hidden)
        output = self.fc(hidden)
        return self.sigmoid(output).squeeze()

# ============== FEATURE NAMES ==============
AUDIO_FEATURE_NAMES = [
    'F0semitoneFrom27.5Hz', 'jitterLocal', 'shimmerLocaldB', 
    'HNRdBACF', 'logRelF0-H1-H2', 'logRelF0-H1-A3', 
    'F1frequency', 'F1bandwidth', 'F1amplitudeLogRelF0',
    'F2frequency', 'F2bandwidth', 'F2amplitudeLogRelF0',
    'F3frequency', 'F3bandwidth', 'F3amplitudeLogRelF0',
    'alphaRatioV', 'hammarbergIndexV', 'slopeV0-500',
    'slopeV500-1500', 'spectralFluxV', 'mfcc1V', 'mfcc2V',
    'mfcc3V', 'mfcc4V', 'loudness'
]

AUDIO_FEATURE_DESCRIPTIONS = {
    'F0semitoneFrom27.5Hz': 'Pitch - Often lower in depression',
    'jitterLocal': 'Voice instability',
    'shimmerLocaldB': 'Amplitude variation',
    'HNRdBACF': 'Voice clarity (lower in depression)',
    'loudness': 'Speech volume (reduced in depression)',
}

# ============== ENHANCED TEXT PREPROCESSING ==============
lemmatizer = WordNetLemmatizer()
try:
    stop_words = set(stopwords.words('english'))
except:
    stop_words = set()

# Keep mental-health-relevant words
KEEP_WORDS = {
    'not', 'no', 'nor', 'never', 'nothing', 'nobody', 'nowhere',
    'neither', 'none', 'cannot', 'without', 'against',
    'very', 'too', 'more', 'most', 'only', 'just',
    'should', 'would', 'could', 'might',
    'myself', 'yourself', 'himself', 'herself',
    'again', 'further', 'why', 'how', 'all', 'each',
    'few', 'own', 'same', 'than', 'until',
    'down', 'out', 'off', 'over', 'under',
}

stop_words = stop_words - KEEP_WORDS

NEGATION_WORDS = {'not', 'no', 'never', 'neither', 'nobody', 'nothing',
                   'nowhere', 'nor', 'cannot', "can't", "won't", "don't",
                   "doesn't", "didn't", "isn't", "aren't", "wasn't", "weren't",
                   "haven't", "hasn't", "hadn't", "wouldn't", "shouldn't",
                   "couldn't", "mustn't", "needn't"}


def preprocess_text(text):
    """Enhanced text preprocessing with negation handling"""
    if pd.isna(text) or not text:
        return ""
    
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    
    # Replace contractions
    contractions = {
        "can't": "cannot", "won't": "will not", "don't": "do not",
        "doesn't": "does not", "didn't": "did not", "isn't": "is not",
        "aren't": "are not", "wasn't": "was not", "weren't": "were not",
        "haven't": "have not", "hasn't": "has not", "hadn't": "had not",
        "wouldn't": "would not", "shouldn't": "should not",
        "couldn't": "could not", "i'm": "i am", "i've": "i have",
        "i'll": "i will", "i'd": "i would", "it's": "it is",
        "that's": "that is", "there's": "there is", "they're": "they are",
        "we're": "we are", "you're": "you are", "he's": "he is",
        "she's": "she is", "what's": "what is", "who's": "who is",
    }
    for contraction, expanded in contractions.items():
        text = text.replace(contraction, expanded)
    
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    try:
        tokens = word_tokenize(text)
    except:
        tokens = text.split()
    
    # Negation handling
    processed_tokens = []
    negate = False
    for token in tokens:
        if token in NEGATION_WORDS or token == 'not':
            negate = True
            processed_tokens.append(token)
            continue
        
        if negate:
            if token in {'but', 'however', 'although', 'though'}:
                negate = False
                processed_tokens.append(token)
            else:
                processed_tokens.append(f'NOT_{token}')
                if len([t for t in processed_tokens if t.startswith('NOT_')]) >= 3:
                    negate = False
        else:
            processed_tokens.append(token)
    
    final_tokens = []
    for token in processed_tokens:
        if token.startswith('NOT_'):
            base = token[4:]
            if len(base) > 2:
                final_tokens.append(f'NOT_{lemmatizer.lemmatize(base)}')
        elif token not in stop_words and len(token) > 2:
            final_tokens.append(lemmatizer.lemmatize(token))
    
    return ' '.join(final_tokens)

# ============== LOADING FUNCTIONS ==============
@st.cache_resource
def load_opensmile():
    """Load OpenSMILE feature extractor"""
    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
    )
    return smile

@st.cache_resource
def get_device():
    """Get the best available device"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')

@st.cache_resource
def load_audio_model():
    """Load the improved audio CNN model"""
    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved_models")
    device = get_device()
    
    # Try loading improved model first
    improved_path = os.path.join(save_dir, "improved_audio_cnn.pth")
    if os.path.exists(improved_path):
        try:
            # Load audio model info for num_channels
            info_path = os.path.join(save_dir, "audio_model_info.pkl")
            if os.path.exists(info_path):
                with open(info_path, 'rb') as f:
                    info = pickle.load(f)
                num_channels = info.get('num_channels', 25)
            else:
                num_channels = 25
            
            model = ImprovedCNNModel(num_channels=num_channels).to(device)
            model.load_state_dict(torch.load(improved_path, map_location=device))
            model.eval()
            return model, 'ImprovedCNN', num_channels
        except Exception as e:
            st.warning(f"Could not load improved audio model: {e}")
    
    # Fallback to simple model
    return None, 'none', 25

@st.cache_resource
def load_tfidf():
    """Load TF-IDF vectorizer"""
    path = os.path.join(save_dir, "tfidf_vectorizer.pkl")
    try:
        with open(path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        st.warning(f"Could not load TF-IDF vectorizer: {e}")
        return None

@st.cache_resource
def load_best_text_model():
    """Load best ML text model (Stacking Ensemble)"""
    path = os.path.join(save_dir, "best_text_model.pkl")
    try:
        with open(path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        st.warning(f"Could not load best text model: {e}")
        return None

@st.cache_resource
def load_ensemble_model():
    """Load ensemble/voting model"""
    # Try the newer voting ensemble first, fallback to old ensemble
    for fname in ['voting_ensemble_model.pkl', 'stacking_ensemble_model.pkl', 'ensemble_model.pkl']:
        path = os.path.join(save_dir, fname)
        try:
            with open(path, 'rb') as f:
                model = pickle.load(f)
            return model
        except Exception:
            continue
    st.warning("Could not load any ensemble model.")
    return None

@st.cache_resource
def load_lstm_model():
    """Load Attention BiLSTM model + vocab — always on CPU to avoid MPS issues"""
    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved_models")
    try:
        with open(f"{save_dir}/lstm_vocab.pkl", 'rb') as f:
            vocab = pickle.load(f)
        with open(f"{save_dir}/model_info.pkl", 'rb') as f:
            info = pickle.load(f)
        
        # Always load on CPU — LSTM is small, CPU inference is fast enough
        max_len = info.get('max_len', 256)
        lstm_type = info.get('lstm_type', 'LSTMClassifier')
        
        if lstm_type == 'AttentionBiLSTM':
            lstm = AttentionBiLSTMClassifier(len(vocab))
        else:
            lstm = LSTMClassifier(len(vocab))
        
        state = torch.load(f"{save_dir}/lstm_model.pth", map_location='cpu')
        lstm.load_state_dict(state)
        lstm.eval()
        return {'lstm': lstm, 'vocab': vocab, 'max_len': max_len, 'lstm_type': lstm_type}
    except Exception as e:
        st.warning(f"Could not load BiLSTM model: {e}")
        return None

@st.cache_resource
def load_feature_importance():
    """Load feature importance for XAI"""
    path = os.path.join(save_dir, "feature_importance.pkl")
    try:
        with open(path, 'rb') as f:
            fi = pickle.load(f)
        # Pre-build lookup dict for O(1) word importance (instead of O(n) list.index)
        if fi and 'feature_names' in fi and 'coefficients' in fi:
            fi['_lookup'] = {name: abs(coef) for name, coef in 
                            zip(fi['feature_names'], fi['coefficients'])}
        return fi
    except:
        return None

def load_text_models():
    """Load all text models (each cached separately for speed)"""
    return {
        'tfidf': load_tfidf(),
        'best_model': load_best_text_model(),
        'ensemble': load_ensemble_model(),
        'feature_importance': load_feature_importance(),
    }

# ============== AUDIO PROCESSING ==============
def extract_audio_features(audio_path, smile):
    """Extract audio features using OpenSMILE"""
    try:
        features = smile.process_file(audio_path)
        numeric_cols = features.select_dtypes(include=[np.number]).columns
        return features[numeric_cols].values
    except Exception as e:
        st.error(f"Error extracting audio features: {e}")
        return None

def create_audio_segments(features, L=128, step=64):
    """Create segments from audio features"""
    segments = []
    epsilon = 1e-8
    
    for start in range(0, len(features) - L + 1, step):
        segment = features[start:start + L]
        std_dev = np.std(segment, axis=0)
        zero_variance = std_dev == 0
        std_dev[zero_variance] = 1
        normalized_segment = (segment - np.mean(segment, axis=0)) / (std_dev + epsilon)
        segments.append(normalized_segment)
    
    return np.array(segments) if segments else None

def predict_audio(model, segments, device):
    """Run audio prediction (batched for speed)"""
    model.eval()
    X = segments.transpose(0, 2, 1)
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        # Process in batches of 64 to avoid memory issues
        batch_size = 64
        all_probs = []
        for i in range(0, len(X_tensor), batch_size):
            batch = X_tensor[i:i+batch_size]
            out = model(batch)
            all_probs.append(out.squeeze().cpu().numpy())
        
        probabilities = np.concatenate(all_probs) if len(all_probs) > 1 else all_probs[0]
        if probabilities.ndim == 0:
            probabilities = np.array([probabilities.item()])
        predictions = (probabilities > 0.5).astype(float)
    
    return predictions, probabilities

def compute_audio_importance(model, segments, device):
    """Compute audio feature importance using gradient saliency (fast — max 16 segments on CPU)"""
    # Sample max 16 segments for speed
    max_samples = 16
    if len(segments) > max_samples:
        idx = np.linspace(0, len(segments) - 1, max_samples, dtype=int)
        sampled = segments[idx]
    else:
        sampled = segments
    
    X = sampled.transpose(0, 2, 1)
    X_tensor = torch.tensor(X, dtype=torch.float32, requires_grad=True)
    
    # Copy weights to a fresh CPU model WITHOUT touching the original cached model
    num_channels = X.shape[1]
    cpu_model = ImprovedCNNModel(num_channels=num_channels)
    cpu_state = {k: v.detach().clone().cpu() for k, v in model.state_dict().items()}
    cpu_model.load_state_dict(cpu_state)
    cpu_model.eval()
    
    cpu_model.zero_grad()
    outputs = cpu_model(X_tensor)
    outputs.sum().backward()
    
    if X_tensor.grad is not None:
        gradients = X_tensor.grad.numpy()
        importance = np.abs(gradients).mean(axis=(0, 2))
    else:
        importance = np.ones(num_channels) / num_channels
    
    return importance

# ============== TEXT PROCESSING ==============
def predict_text_tfidf(text, tfidf, model):
    """Predict using TF-IDF + ML model"""
    try:
        cleaned = preprocess_text(text)
        if not cleaned:
            st.warning("Text preprocessing returned empty — try a longer input.")
            return None, None
        
        features = tfidf.transform([cleaned])
        prediction = model.predict(features)[0]
        
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(features)[0][1]
        else:
            proba = 0.5 + (0.5 if prediction == 1 else -0.5)
        
        return int(prediction), float(proba)
    except Exception as e:
        st.error(f"Text prediction failed: {e}")
        return None, None

def predict_text_lstm(text, vocab, model, device, max_len=256):
    """Predict using Attention BiLSTM (model always on CPU)"""
    try:
        cleaned = preprocess_text(text)
        if not cleaned:
            st.warning("Text preprocessing returned empty — try a longer input.")
            return None, None
        
        tokens = cleaned.split()[:max_len]
        indices = [vocab.get(token, 1) for token in tokens]
        
        if len(indices) < max_len:
            indices += [0] * (max_len - len(indices))
        
        # Tensor on CPU — model is loaded on CPU in load_lstm_model()
        X = torch.tensor([indices], dtype=torch.long)
        
        model.eval()
        with torch.no_grad():
            output = model(X)   # model stays on CPU, no .cpu() call needed
            proba = float(output.item())
            prediction = 1 if proba > 0.5 else 0
        
        return prediction, proba
    except Exception as e:
        st.error(f"BiLSTM prediction failed: {e}")
        return None, None

def get_text_word_importance(text, tfidf, feature_importance):
    """Get word-level importance for XAI (uses pre-built O(1) lookup)"""
    if not feature_importance:
        return {}
    
    cleaned = preprocess_text(text)
    words = cleaned.split()
    
    # Use pre-built lookup if available (O(1) per word)
    lookup = feature_importance.get('_lookup')
    if lookup:
        return {w: lookup[w] for w in words if w in lookup}
    
    # Fallback to original method
    feature_names = feature_importance['feature_names']
    coefficients = np.array(feature_importance['coefficients'])
    word_importance = {}
    for word in words:
        if word in feature_names:
            idx = feature_names.index(word)
            word_importance[word] = abs(coefficients[idx])
    
    return word_importance

# ============== FUSION ==============
def fuse_predictions(audio_pred, audio_conf, text_pred, text_conf, 
                     audio_weight=0.5, text_weight=0.5):
    """Fuse audio and text predictions with calibrated confidence"""
    if audio_pred is None and text_pred is None:
        return None, None
    
    if audio_pred is None:
        return text_pred, text_conf
    if text_pred is None:
        return audio_pred, audio_conf
    
    # Weighted average of confidences
    fused_conf = (audio_conf * audio_weight + text_conf * text_weight)
    fused_pred = 1 if fused_conf > 0.5 else 0
    
    return fused_pred, fused_conf

# ============== VISUALIZATIONS ==============
def create_modality_comparison_chart(audio_conf, text_conf, fused_conf):
    """Create comparison chart for modalities"""
    fig = go.Figure()
    
    categories = ['Audio', 'Text', 'Fused']
    values = [audio_conf * 100 if audio_conf else 0, 
              text_conf * 100 if text_conf else 0, 
              fused_conf * 100 if fused_conf else 0]
    colors = ['#667eea', '#38ef7d', '#ff6b6b']
    
    fig.add_trace(go.Bar(
        x=categories,
        y=values,
        marker_color=colors,
        text=[f"{v:.1f}%" for v in values],
        textposition='outside'
    ))
    
    fig.add_hline(y=50, line_dash="dash", line_color="red", 
                  annotation_text="Threshold")
    
    fig.update_layout(
        title="Detection Confidence by Modality",
        yaxis_title="Confidence %",
        yaxis_range=[0, 100],
        height=350,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(245,245,245,1)'
    )
    
    return fig

def create_feature_importance_chart(importance, names, title):
    """Create feature importance bar chart"""
    sorted_idx = np.argsort(importance)[::-1][:15]
    
    fig = go.Figure(go.Bar(
        x=importance[sorted_idx],
        y=[names[i] if i < len(names) else f"F{i}" for i in sorted_idx],
        orientation='h',
        marker=dict(
            color=importance[sorted_idx],
            colorscale='Viridis'
        )
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Importance",
        height=400,
        yaxis=dict(autorange="reversed"),
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

def create_word_cloud_data(word_importance):
    """Create word importance visualization"""
    if not word_importance:
        return None
    
    sorted_words = sorted(word_importance.items(), key=lambda x: x[1], reverse=True)[:20]
    
    fig = go.Figure(go.Bar(
        x=[w[1] for w in sorted_words],
        y=[w[0] for w in sorted_words],
        orientation='h',
        marker=dict(
            color=[w[1] for w in sorted_words],
            colorscale='Reds'
        )
    ))
    
    fig.update_layout(
        title="Key Words Contributing to Prediction",
        xaxis_title="Importance",
        height=400,
        yaxis=dict(autorange="reversed"),
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

# ============== MAIN APP ==============
st.markdown('<h1 class="main-header">🧠 Multimodal Depression Detection</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Audio + Text Analysis with Explainable AI (Enhanced Models)</p>', unsafe_allow_html=True)

# Load model info for metrics display
save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved_models")
text_model_info = None
audio_model_info = None
try:
    with open(f"{save_dir}/model_info.pkl", 'rb') as f:
        text_model_info = pickle.load(f)
except:
    pass
try:
    with open(f"{save_dir}/audio_model_info.pkl", 'rb') as f:
        audio_model_info = pickle.load(f)
except:
    pass

# Sidebar
with st.sidebar:
    st.markdown("## 🎯 About")
    st.markdown("""
    This app combines **two modalities** for depression detection:
    - 🎤 **Audio**: Improved CNN analyzes speech patterns
    - 📝 **Text**: Ensemble + Attention BiLSTM analyzes written content
    
    Both use **XAI** to explain predictions.
    """)
    
    st.markdown("---")
    st.markdown("### ⚙️ Fusion Settings")
    
    fusion_method = st.selectbox(
        "Fusion Method",
        ["Weighted Average", "Audio Priority", "Text Priority", "Max Confidence"]
    )
    
    if fusion_method == "Weighted Average":
        audio_weight = st.slider("Audio Weight", 0.0, 1.0, 0.5, 0.1)
        text_weight = 1.0 - audio_weight
        st.write(f"Text Weight: {text_weight:.1f}")
    elif fusion_method == "Audio Priority":
        audio_weight, text_weight = 0.7, 0.3
    elif fusion_method == "Text Priority":
        audio_weight, text_weight = 0.3, 0.7
    else:
        audio_weight, text_weight = 0.5, 0.5
    
    st.markdown("---")
    st.markdown("### 📊 Model Performance")
    
    # Display real metrics if available
    if audio_model_info:
        st.metric("Audio CNN Acc", f"{audio_model_info.get('file_accuracy', 0)*100:.1f}%")
        st.metric("Audio CNN F1", f"{audio_model_info.get('file_f1', 0)*100:.1f}%")
    else:
        st.metric("Audio Model", "Not trained yet")
    
    if text_model_info:
        best_name = text_model_info.get('best_model_name', 'Unknown')
        best_metrics = text_model_info.get('metrics', {}).get(best_name, {})
        if best_metrics:
            st.metric(f"Text ({best_name}) Acc", f"{best_metrics.get('Accuracy', 0)*100:.1f}%")
            st.metric(f"Text ({best_name}) F1", f"{best_metrics.get('F1 Score', 0)*100:.1f}%")
    else:
        st.metric("Text Model", "Not trained yet")
    
    st.markdown("---")
    st.markdown("### 🏗️ Model Architecture")
    st.markdown("""
    **Audio**: 3-layer CNN + BatchNorm + GAP  
    **Text ML**: GridSearch-tuned ensemble  
    **Text DL**: Attention BiLSTM  
    **Fusion**: Weighted confidence fusion
    """)

# Main content tabs
tab1, tab2, tab3 = st.tabs(["🎤 Audio Analysis", "📝 Text Analysis", "🔀 Multimodal Fusion"])

# Initialize session state
if 'audio_result' not in st.session_state:
    st.session_state.audio_result = None
if 'text_result' not in st.session_state:
    st.session_state.text_result = None

# ============== AUDIO TAB ==============
with tab1:
    st.markdown("### 🎤 Audio-Based Depression Detection")
    st.markdown("*Using Improved 3-Layer CNN with BatchNorm*")
    
    uploaded_audio = st.file_uploader(
        "Upload a WAV audio file",
        type=['wav'],
        key='audio_upload'
    )
    
    if uploaded_audio:
        col1, col2 = st.columns([2, 1])
        with col1:
            st.audio(uploaded_audio, format='audio/wav')
        with col2:
            st.metric("File", uploaded_audio.name[:20] + "...")
            st.metric("Size", f"{uploaded_audio.size / 1024:.1f} KB")
        
        if st.button("🔍 Analyze Audio", key='analyze_audio'):
            with st.spinner("Processing audio with Improved CNN..."):
                # Save temp file
                with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
                    tmp.write(uploaded_audio.getvalue())
                    tmp_path = tmp.name
                
                # Extract features
                smile = load_opensmile()
                features = extract_audio_features(tmp_path, smile)
                
                if features is not None:
                    segments = create_audio_segments(features)
                    
                    if segments is not None:
                        device = get_device()
                        
                        # Try loading improved model
                        audio_model, model_type, num_ch = load_audio_model()
                        
                        if audio_model is None:
                            # Fallback to creating a new model
                            model_type = 'SimpleCNN'
                            audio_model = AudioCNNModel(segments.shape[2]).to(device)
                        
                        predictions, probabilities = predict_audio(audio_model, segments, device)
                        importance = compute_audio_importance(audio_model, segments, device)
                        
                        # Aggregate
                        vote = Counter(predictions).most_common(1)[0][0]
                        avg_conf = np.mean(probabilities)
                        
                        # Store result
                        st.session_state.audio_result = {
                            'prediction': int(vote),
                            'confidence': avg_conf,
                            'importance': importance,
                            'segments': len(segments),
                            'model_type': model_type
                        }
                        
                        # Display results
                        st.markdown("---")
                        st.markdown("### 📈 Audio Analysis Results")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Segments Analyzed", len(segments))
                        with col2:
                            st.metric("Confidence", f"{avg_conf:.1%}")
                        with col3:
                            result_text = "Depression Detected" if vote == 1 else "No Depression"
                            st.metric("Result", result_text)
                        with col4:
                            st.metric("Model", model_type)
                        
                        # Feature importance
                        st.markdown("#### 🔍 Audio Feature Importance")
                        fig = create_feature_importance_chart(
                            importance, AUDIO_FEATURE_NAMES,
                            "Top Audio Features (Gradient Saliency)"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                
                os.unlink(tmp_path)

# ============== TEXT TAB ==============
with tab2:
    st.markdown("### 📝 Text-Based Depression Detection")
    st.markdown("*Using Enhanced NLP with Negation Handling*")
    
    text_input = st.text_area(
        "Enter text to analyze (diary entry, social media post, etc.)",
        height=200,
        placeholder="Type or paste text here..."
    )
    
    # Determine available model choices
    model_choices = ["Best ML Model (Tuned)", "Ensemble (Soft Voting)", "Attention BiLSTM"]
    text_model_choice = st.radio(
        "Select Model",
        model_choices,
        horizontal=True
    )
    
    if st.button("🔍 Analyze Text", key='analyze_text'):
        if text_input.strip():
            with st.spinner("Analyzing text..."):
                device = get_device()
                
                pred, conf = None, None
                used_model = text_model_choice
                
                if text_model_choice == "Best ML Model (Tuned)":
                    tfidf = load_tfidf()
                    best_model = load_best_text_model()
                    if tfidf and best_model:
                        pred, conf = predict_text_tfidf(text_input, tfidf, best_model)
                    else:
                        st.error("Text model not loaded. Please train the model first.")
                
                elif text_model_choice == "Ensemble (Soft Voting)":
                    tfidf = load_tfidf()
                    ensemble = load_ensemble_model()
                    if tfidf and ensemble:
                        pred, conf = predict_text_tfidf(text_input, tfidf, ensemble)
                    elif tfidf:
                        best_model = load_best_text_model()
                        if best_model:
                            st.warning("Ensemble not available, falling back to best ML model.")
                            pred, conf = predict_text_tfidf(text_input, tfidf, best_model)
                            used_model = "Best ML Model (fallback)"
                    else:
                        st.error("No text models loaded.")
                
                else:  # Attention BiLSTM
                    lstm_data = load_lstm_model()
                    if lstm_data:
                        pred, conf = predict_text_lstm(
                            text_input,
                            lstm_data['vocab'],
                            lstm_data['lstm'],
                            device,
                            max_len=lstm_data.get('max_len', 256)
                        )
                    else:
                        st.error("LSTM model not loaded.")
                
                if pred is not None:
                    # Store result
                    st.session_state.text_result = {
                        'prediction': pred,
                        'confidence': conf,
                        'text': text_input,
                        'model_used': used_model
                    }
                    
                    # Display results
                    st.markdown("---")
                    st.markdown("### 📈 Text Analysis Results")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        severity = "Severe (Suicidal Indicators)" if pred == 1 else "Non-Severe Depression"
                        st.metric("Severity Level", severity)
                    with col2:
                        st.metric("Confidence", f"{conf:.1%}")
                    with col3:
                        st.metric("Model Used", used_model.split('(')[0].strip())
                    
                    # Word importance
                    fi = load_feature_importance()
                    if fi:
                        word_imp = get_text_word_importance(text_input, None, fi)
                        if word_imp:
                            st.markdown("#### 🔍 Key Words Analysis")
                            fig = create_word_cloud_data(word_imp)
                            if fig:
                                st.plotly_chart(fig, use_container_width=True)
                    
                    # Explanation
                    st.markdown("#### 💡 Interpretation")
                    if pred == 1:
                        st.warning("""
                        ⚠️ **Severe depression/suicidal indicators detected**
                        
                        The text contains language patterns associated with:
                        - Expressions of hopelessness
                        - Self-harm or suicidal ideation
                        - Deep emotional distress
                        """)
                    else:
                        st.info("""
                        ℹ️ **Depression indicators detected (non-severe)**
                        
                        The text contains language patterns associated with:
                        - General depressive symptoms
                        - Emotional struggles
                        - Mental health concerns
                        """)
        else:
            st.warning("Please enter some text to analyze.")

# ============== MULTIMODAL TAB ==============
with tab3:
    st.markdown("### 🔀 Multimodal Fusion Analysis")
    
    st.markdown("""
    This tab combines both audio and text analyses for a more comprehensive assessment.
    **Instructions:**
    1. First analyze audio in the Audio tab
    2. Then analyze text in the Text tab
    3. Come back here to see the fused result
    """)
    
    audio_result = st.session_state.audio_result
    text_result = st.session_state.text_result
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🎤 Audio Status")
        if audio_result:
            st.success(f"✅ Analyzed ({audio_result['segments']} segments) — {audio_result.get('model_type', 'CNN')}")
            st.write(f"Prediction: {'Depression' if audio_result['prediction'] == 1 else 'No Depression'}")
            st.write(f"Confidence: {audio_result['confidence']:.1%}")
        else:
            st.info("⏳ No audio analyzed yet")
    
    with col2:
        st.markdown("#### 📝 Text Status")
        if text_result:
            st.success(f"✅ Analyzed — {text_result.get('model_used', 'ML Model')}")
            st.write(f"Prediction: {'Severe' if text_result['prediction'] == 1 else 'Non-Severe'}")
            st.write(f"Confidence: {text_result['confidence']:.1%}")
        else:
            st.info("⏳ No text analyzed yet")
    
    st.markdown("---")
    
    if audio_result or text_result:
        st.markdown("### 🎯 Fused Prediction")
        
        audio_pred = audio_result['prediction'] if audio_result else None
        audio_conf = audio_result['confidence'] if audio_result else None
        text_pred = text_result['prediction'] if text_result else None
        text_conf = text_result['confidence'] if text_result else None
        
        # Apply fusion
        if fusion_method == "Max Confidence":
            if audio_conf and text_conf:
                if audio_conf > text_conf:
                    fused_pred, fused_conf = audio_pred, audio_conf
                else:
                    fused_pred, fused_conf = text_pred, text_conf
            else:
                fused_pred, fused_conf = fuse_predictions(
                    audio_pred, audio_conf, text_pred, text_conf
                )
        else:
            fused_pred, fused_conf = fuse_predictions(
                audio_pred, audio_conf, text_pred, text_conf,
                audio_weight, text_weight
            )
        
        # Display fused result
        if fused_pred is not None:
            col1, col2 = st.columns([1, 1])
            
            with col1:
                if fused_pred == 1:
                    st.markdown("""
                    <div class="prediction-positive">
                        <h2>⚠️ Depression Detected</h2>
                        <p>Multimodal analysis indicates signs of depression</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="prediction-negative">
                        <h2>✅ No Depression Detected</h2>
                        <p>Multimodal analysis shows normal patterns</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            with col2:
                # Confidence gauge
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=fused_conf * 100,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Fused Confidence"},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "#667eea"},
                        'steps': [
                            {'range': [0, 50], 'color': "#e0e0e0"},
                            {'range': [50, 75], 'color': "#ffeaa7"},
                            {'range': [75, 100], 'color': "#ff7675"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 50
                        }
                    }
                ))
                fig.update_layout(height=250, paper_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig, use_container_width=True)
            
            # Comparison chart
            st.markdown("### 📊 Modality Comparison")
            fig = create_modality_comparison_chart(audio_conf, text_conf, fused_conf)
            st.plotly_chart(fig, use_container_width=True)
            
            # Explanation
            st.markdown("### 💡 Multimodal Explanation")
            
            explanation_parts = []
            if audio_result:
                audio_text = "detected depression indicators in speech patterns (voice pitch, clarity, volume changes)"
                if audio_result['prediction'] == 0:
                    audio_text = "found normal speech patterns without depression indicators"
                explanation_parts.append(f"**Audio Analysis** ({audio_result.get('model_type', 'CNN')}): {audio_text}")
            
            if text_result:
                text_text = "identified severe depression/suicidal language patterns"
                if text_result['prediction'] == 0:
                    text_text = "found mild depression indicators in text"
                explanation_parts.append(f"**Text Analysis** ({text_result.get('model_used', 'ML')}): {text_text}")
            
            fusion_text = f"**Fusion ({fusion_method})**: Combined both modalities with audio weight {audio_weight:.0%} and text weight {text_weight:.0%}"
            explanation_parts.append(fusion_text)
            
            for part in explanation_parts:
                st.markdown(part)
    else:
        st.info("👆 Please analyze audio and/or text in the other tabs first.")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>⚠️ <strong>Disclaimer:</strong> This tool is for research and educational purposes only. 
    It should not be used as a substitute for professional medical diagnosis.</p>
    <p>Built with ❤️ using Streamlit, PyTorch, OpenSMILE & NLP | Enhanced Models v2</p>
</div>
""", unsafe_allow_html=True)
