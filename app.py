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
# Audio processing — optional (may fail on some cloud environments)
try:
    import opensmile
    OPENSMILE_AVAILABLE = True
except Exception:
    OPENSMILE_AVAILABLE = False
import os
import tempfile
import pickle
import re
from collections import Counter
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# LLM — optional (silently disabled if no API key)
try:
    import openai
    OPENAI_AVAILABLE = True
except Exception:
    OPENAI_AVAILABLE = False

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
    page_title="MiSec · Multimodal Depression Detection",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# DESIGN SYSTEM CSS — Supports both Light Landing & Dark Dashboard
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=Inter:wght@400;500;600;700&display=swap');

/* ---- HIDE STREAMLIT CHROME ---- */
#MainMenu {visibility: hidden !important;}
header {visibility: hidden !important;}
footer {visibility: hidden !important;}

/* ---- LIGHT THEME TOKENS (Landing) ---- */
:root {
    --l-bg: #f8f7fa;
    --l-surface: #ffffff;
    --l-fg: #1a1625;
    --l-muted: #6b6680;
    --l-border: #e8e5f0;
    --l-accent: #6d5acf;
    --grad-start: #667eea;
    --grad-end: #764ba2;
    --radius: 10px;
    --radius-lg: 16px;
    --radius-xl: 20px;
}

/* ---- DARK THEME TOKENS (Dashboard) ---- */
:root {
    --d-bg: #0d1117;
    --d-surface: #161b22;
    --d-surface-2: #21262d;
    --d-fg: #e6edf3;
    --d-muted: #8b949e;
    --d-border: #30363d;
    --d-accent: #a371f7;
    --d-accent-2: #58a6ff;
    --d-danger: #f85149;
    --d-success: #3fb950;
    --d-warn: #d29922;
}

.stApp { background: var(--l-bg); }
.block-container { padding-top: 1rem; padding-bottom: 2rem; }

/* ---- TYPOGRAPHY ---- */
.h1, .h1-misec, .stMarkdown h1, [data-testid="stMarkdownContainer"] h1 {
    font-family: 'DM Serif Display', Georgia, serif;
    font-size: clamp(36px, 5vw, 64px);
    line-height: 1.04;
    letter-spacing: -0.02em;
    color: var(--l-fg) !important;
    margin: 0;
}
.h2, .h2-misec, .stMarkdown h2, [data-testid="stMarkdownContainer"] h2 {
    font-family: 'DM Serif Display', Georgia, serif;
    font-size: clamp(26px, 3.5vw, 40px);
    line-height: 1.1;
    letter-spacing: -0.015em;
    color: var(--l-fg) !important;
    margin: 0;
}
.h3, .h3-misec, .stMarkdown h3, [data-testid="stMarkdownContainer"] h3 {
    font-size: 20px;
    font-weight: 600;
    line-height: 1.3;
    color: var(--l-fg) !important;
    margin: 0;
}
.lead-misec {
    font-size: 17px;
    line-height: 1.55;
    color: var(--l-muted);
    max-width: 60ch;
    margin: 0;
}
.eyebrow {
    font-family: ui-monospace, 'SF Mono', Menlo, monospace;
    font-size: 11px;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--l-accent);
    margin: 0 0 12px;
}
.meta-misec {
    font-family: ui-monospace, 'SF Mono', Menlo, monospace;
    font-size: 13px;
    color: var(--l-muted);
}

/* ---- BUTTONS ---- */
.btn-primary-misec {
    display: inline-flex; align-items: center; gap: 8px;
    padding: 14px 32px; border-radius: var(--radius);
    background: linear-gradient(135deg, var(--grad-start), var(--grad-end));
    color: white; border: none; font-size: 15px; font-weight: 500;
    cursor: pointer; text-decoration: none;
    box-shadow: 0 4px 14px rgba(102,126,234,0.25);
    transition: box-shadow .2s, filter .2s;
}
.btn-primary-misec:hover { box-shadow: 0 6px 20px rgba(102,126,234,0.35); filter: brightness(1.05); }
.btn-secondary-misec {
    display: inline-flex; align-items: center; gap: 8px;
    padding: 14px 32px; border-radius: var(--radius);
    background: transparent; color: var(--l-fg); border: 1px solid var(--l-border);
    font-size: 15px; font-weight: 500; cursor: pointer; text-decoration: none;
    transition: border-color .2s, background .2s;
}
.btn-secondary-misec:hover { border-color: var(--l-fg); background: rgba(26,22,37,0.04); }
.btn-white {
    background: white !important; color: var(--grad-end) !important;
    box-shadow: 0 4px 14px rgba(0,0,0,0.15) !important;
}
.btn-white:hover { box-shadow: 0 6px 20px rgba(0,0,0,0.2) !important; }

/* ---- CARDS (Light) ---- */
.card-misec {
    background: var(--l-surface); border: 1px solid var(--l-border);
    border-radius: var(--radius-lg); padding: 28px;
    transition: transform .2s, box-shadow .2s;
}
.card-misec:hover { transform: translateY(-2px); box-shadow: 0 8px 24px rgba(26,22,37,0.06); }

/* ---- HERO ---- */
.hero-section {
    background: linear-gradient(135deg, var(--grad-start), var(--grad-end));
    color: white; padding: clamp(48px,8vw,96px) 2rem;
    border-radius: var(--radius-lg); position: relative; overflow: hidden;
    margin-bottom: 2rem;
}
.hero-section::before {
    content: ''; position: absolute; inset: 0;
    background: radial-gradient(circle at 20% 80%, rgba(255,255,255,0.12) 0%, transparent 50%),
                radial-gradient(circle at 80% 20%, rgba(255,255,255,0.08) 0%, transparent 40%);
    pointer-events: none;
}
.hero-section .eyebrow { color: rgba(255,255,255,0.85); position: relative; z-index: 1; }
.hero-section h1 { color: white !important; margin-bottom: 20px; position: relative; z-index: 1; }
.hero-section .lead-misec { color: rgba(255,255,255,0.85); margin: 0 auto 32px; max-width: 52ch; position: relative; z-index: 1; }

/* ---- DARK SECTIONS (Landing) ---- */
.dark-section {
    background: var(--d-bg); color: var(--d-fg);
    border-radius: var(--radius-lg); padding: clamp(48px,6vw,80px) 2rem;
    margin: 2rem 0;
}
.dark-section h2, .dark-section h3 { color: var(--d-fg) !important; }
.dark-section .lead-misec { color: var(--d-muted); }

/* ---- NAVBAR ---- */
.nav-bar {
    display: flex; align-items: center; justify-content: space-between;
    padding: 16px 2rem; max-width: 1200px; margin: 0 auto;
}
.nav-logo { font-family: 'DM Serif Display', Georgia, serif; font-size: 24px; color: var(--l-fg); font-weight: 600; }
.nav-links { display: flex; gap: 32px; align-items: center; }
.nav-links a { color: var(--l-muted); text-decoration: none; font-size: 14px; font-weight: 500; transition: color .2s; }
.nav-links a:hover { color: var(--l-fg); }

/* ---- STATS ---- */
.stat-card-misec { text-align: center; padding: 2rem 1rem; }
.stat-num-misec {
    font-family: 'DM Serif Display', Georgia, serif;
    font-size: clamp(48px,7vw,80px); line-height: 0.95; letter-spacing: -0.04em;
    color: var(--l-accent); font-weight: 400;
}
.stat-label-misec { color: var(--l-muted); font-size: 15px; margin-top: 10px; max-width: 28ch; line-height: 1.45; margin-inline: auto; }
.stat-unit { font-size: 0.5em; opacity: 0.7; margin-left: 2px; }

/* ---- STEPS ---- */
.step-card { text-align: center; padding: 32px 24px; }
.step-num {
    width: 48px; height: 48px; border-radius: 50%;
    background: linear-gradient(135deg, var(--grad-start), var(--grad-end));
    color: white; display: grid; place-items: center;
    font-weight: 700; font-size: 18px; margin: 0 auto 16px;
}
.step-title { font-weight: 600; font-size: 18px; color: var(--l-fg); margin-bottom: 8px; }
.step-desc { color: var(--l-muted); font-size: 14px; line-height: 1.6; }

/* ---- FEATURE ICON ---- */
.feature-mark {
    width: 44px; height: 44px; display: grid; place-items: center;
    border: 1px solid var(--l-border); border-radius: 12px;
    color: var(--l-accent); margin-bottom: 16px;
    background: var(--l-surface); font-size: 20px;
}

/* ---- PERFORMANCE ---- */
.perf-card { text-align: center; padding: 40px 28px; background: var(--l-surface); border: 1px solid var(--l-border); border-radius: var(--radius-lg); }
.perf-num {
    font-family: 'DM Serif Display', Georgia, serif;
    font-size: clamp(36px,4vw,56px); line-height: 1; letter-spacing: -0.03em;
    background: linear-gradient(135deg, var(--grad-start), var(--grad-end));
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.perf-label { color: var(--l-muted); font-size: 14px; margin-top: 8px; }
.perf-detail { color: var(--l-fg); font-size: 13px; margin-top: 4px; font-weight: 500; }

/* ---- BADGE PILLS ---- */
.badge-pill {
    display: inline-block; padding: 6px 14px; border-radius: 999px;
    font-size: 12px; font-weight: 500; margin: 4px;
}
.badge-red { background: rgba(248,81,73,0.12); color: var(--d-danger); }
.badge-green { background: rgba(63,185,80,0.12); color: var(--d-success); }
.badge-purple { background: rgba(163,113,247,0.12); color: var(--d-accent); }
.badge-blue { background: rgba(88,166,255,0.12); color: var(--d-accent-2); }
.badge-amber { background: rgba(210,153,34,0.12); color: var(--d-warn); }

/* ---- TESTIMONIAL ---- */
.testi-card {
    background: var(--l-surface); border: 1px solid var(--l-border);
    border-radius: var(--radius-lg); padding: 28px;
}
.testi-quote { font-style: italic; color: var(--l-fg); font-size: 15px; line-height: 1.6; margin-bottom: 16px; }
.testi-author { display: flex; align-items: center; gap: 12px; }
.testi-avatar {
    width: 40px; height: 40px; border-radius: 50%;
    background: linear-gradient(135deg, var(--grad-start), var(--grad-end));
    color: white; display: grid; place-items: center;
    font-weight: 600; font-size: 14px;
}
.testi-name { font-weight: 600; font-size: 14px; color: var(--l-fg); }
.testi-role { font-size: 12px; color: var(--l-muted); }

/* ---- CTA SECTION ---- */
.cta-section {
    background: linear-gradient(135deg, var(--grad-start), var(--grad-end));
    color: white; text-align: center;
    padding: clamp(48px,6vw,80px) 2rem; border-radius: var(--radius-lg);
    position: relative; overflow: hidden; margin: 2rem 0;
}
.cta-section::before {
    content: ''; position: absolute; inset: 0;
    background: radial-gradient(circle at 50% 0%, rgba(255,255,255,0.1) 0%, transparent 60%);
    pointer-events: none;
}
.cta-section h2 { color: white !important; position: relative; z-index: 1; }
.cta-section .lead-misec { color: rgba(255,255,255,0.85); margin: 16px auto 32px; position: relative; z-index: 1; max-width: 52ch; }

/* ---- FOOTER ---- */
.pagefoot {
    text-align: center; color: var(--l-muted); font-size: 13px;
    padding: 2rem; border-top: 1px solid var(--l-border); margin-top: 2rem;
}

/* ---- APP DARK THEME OVERRIDES ---- */
.app-dark .stApp { background: var(--d-bg) !important; }
.app-dark h1, .app-dark h2, .app-dark h3,
.app-dark .stMarkdown h1, .app-dark .stMarkdown h2, .app-dark .stMarkdown h3,
.app-dark [data-testid="stMarkdownContainer"] h1,
.app-dark [data-testid="stMarkdownContainer"] h2,
.app-dark [data-testid="stMarkdownContainer"] h3 {
    color: var(--d-fg) !important;
}
.app-dark p, .app-dark li, .app-dark span, .app-dark label {
    color: var(--d-muted) !important;
}
.app-dark .card-misec {
    background: var(--d-surface) !important; border-color: var(--d-border) !important;
}

/* Dark sidebar */
.app-dark [data-testid="stSidebar"] {
    background: var(--d-surface) !important; border-right-color: var(--d-border) !important;
}
.app-dark [data-testid="stSidebar"] p,
.app-dark [data-testid="stSidebar"] li,
.app-dark [data-testid="stSidebar"] span,
.app-dark [data-testid="stSidebar"] .stMarkdown,
.app-dark [data-testid="stSidebar"] strong,
.app-dark [data-testid="stSidebar"] label {
    color: var(--d-fg) !important;
}
.app-dark [data-testid="stSidebar"] [data-testid="stMetricValue"] {
    color: var(--d-fg) !important;
}
.app-dark [data-testid="stSidebar"] [data-testid="stMetricLabel"] {
    color: var(--d-muted) !important;
}

/* Dark tabs */
.app-dark .stTabs [data-baseweb="tab"] {
    color: var(--d-muted) !important; background: transparent !important;
}
.app-dark .stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, var(--grad-start), var(--grad-end)) !important;
    color: white !important;
}

/* Dark buttons */
.app-dark     [data-testid="stMarkdownContainer"] h1,
    [data-testid="stMarkdownContainer"] h2,
    [data-testid="stMarkdownContainer"] h3,
    [data-testid="stMarkdownContainer"] h4 {
        color: #e6edf3 !important;
    }
    .stButton > button[kind="secondary"] {
    background: var(--d-surface-2) !important; color: var(--d-fg) !important;
    border-color: var(--d-border) !important;
}
.app-dark .stButton > button[kind="secondary"]:hover {
    border-color: var(--d-accent) !important; background: var(--d-surface) !important;
}

/* Dark file uploader */
.app-dark [data-testid="stFileUploader"] > section {
    border-color: var(--d-border) !important; background: var(--d-surface) !important;
}
.app-dark [data-testid="stFileUploader"] > section:hover {
    border-color: var(--d-accent) !important;
}
.app-dark [data-testid="stFileUploader"] span {
    color: var(--d-fg) !important;
}

/* Dark text area / inputs */
.app-dark textarea, .app-dark input, .app-dark .stTextArea textarea,
.app-dark [data-baseweb="textarea"] textarea {
    background: var(--d-surface) !important; color: var(--d-fg) !important;
    border-color: var(--d-border) !important;
}
.app-dark [data-testid="stSelectbox"] > div[data-baseweb="select"] {
    background: var(--d-surface) !important; color: var(--d-fg) !important;
    border-color: var(--d-border) !important;
}
.app-dark [data-testid="stSlider"] [data-testid="stThumbValue"] {
    color: var(--d-fg) !important;
}
.app-dark [data-testid="stRadio"] label {
    color: var(--d-fg) !important;
}

/* Dark metric */
.app-dark [data-testid="stMetricValue"] { color: var(--d-fg) !important; }
.app-dark [data-testid="stMetricLabel"] { color: var(--d-muted) !important; }
.app-dark [data-testid="stMetricDelta"] { color: var(--d-accent) !important; }

/* Dark expander */
.app-dark [data-testid="stExpander"] {
    background: var(--d-surface) !important; border-color: var(--d-border) !important;
}

/* Streamlit button override (global) */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, var(--grad-start), var(--grad-end)) !important;
    color: white !important; border: none !important;
    border-radius: var(--radius) !important; font-weight: 500 !important;
    box-shadow: 0 4px 14px rgba(102,126,234,0.25) !important;
}
.stButton > button[kind="primary"]:hover {
    box-shadow: 0 6px 20px rgba(102,126,234,0.35) !important;
    filter: brightness(1.05) !important;
}

/* Tabs global */
.stTabs [data-baseweb="tab-list"] { gap: 8px; }
.stTabs [data-baseweb="tab"] {
    border-radius: var(--radius) var(--radius) 0 0; padding: 10px 20px;
    font-weight: 500; color: var(--l-fg) !important; background: transparent !important;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, var(--grad-start), var(--grad-end)) !important;
    color: white !important;
}

/* Sidebar global */
[data-testid="stSidebar"] { background: var(--l-surface); border-right: 1px solid var(--l-border); }
[data-testid="stSidebar"] .block-container { padding-top: 2rem; }
[data-testid="stSidebar"] p, [data-testid="stSidebar"] li,
[data-testid="stSidebar"] span, [data-testid="stSidebar"] .stMarkdown {
    color: var(--l-fg) !important;
}
[data-testid="stSidebar"] strong { color: var(--l-fg) !important; font-weight: 600; }

/* File uploader global */
[data-testid="stFileUploader"] > section {
    border: 2px dashed var(--l-border); border-radius: var(--radius-lg); background: var(--l-surface);
}
[data-testid="stFileUploader"] > section:hover { border-color: var(--l-accent); }

/* Dashboard metric cards (dark) */
.dash-metric-card {
    background: var(--d-surface); border: 1px solid var(--d-border);
    border-radius: var(--radius-lg); padding: 24px;
}
.dash-metric-label { font-size: 12px; color: var(--d-muted); text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 8px; }
.dash-metric-value { font-size: 32px; font-weight: 700; color: var(--d-fg); }
.dash-metric-value.danger { color: var(--d-danger); }
.dash-metric-value.success { color: var(--d-success); }
.dash-metric-sub { font-size: 12px; color: var(--d-muted); margin-top: 4px; }

/* Dashboard nav sidebar */
.dash-nav-item {
    display: flex; align-items: center; gap: 12px;
    padding: 10px 16px; border-radius: var(--radius);
    color: var(--d-muted); font-size: 14px; cursor: pointer;
    transition: background .2s, color .2s; margin-bottom: 4px;
}
.dash-nav-item:hover { background: var(--d-surface-2); color: var(--d-fg); }
.dash-nav-item.active { background: var(--d-surface-2); color: var(--d-accent); font-weight: 500; }

/* Prediction cards (dark) */
.pred-card-dark {
    background: var(--d-surface); border: 1px solid var(--d-border);
    border-radius: var(--radius-lg); padding: 32px; text-align: center;
}
.pred-card-dark.positive { border-color: var(--d-danger); }
.pred-card-dark.negative { border-color: var(--d-success); }
.pred-card-dark h2 { margin-bottom: 8px; }

/* Sparkline bar (for demo) */
.sparkline { display: flex; align-items: flex-end; gap: 2px; height: 40px; }
.spark-bar { flex: 1; background: var(--d-accent); border-radius: 2px; opacity: 0.7; }
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
    if not OPENSMILE_AVAILABLE:
        st.error("OpenSMILE is not available. Audio analysis is disabled.")
        return None
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

# ============== LLM DEEP ANALYSIS ==============
def generate_llm_analysis(audio_result, text_result, fusion_method, audio_weight, text_weight, raw_text=""):
    """Call OpenAI GPT-4o-mini for an in-depth multimodal clinical analysis.
    Runs entirely in the backend. Returns the LLM response text or None.
    """
    if not OPENAI_AVAILABLE:
        return None
    
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return None
    
    client = openai.OpenAI(api_key=api_key)
    
    # Build audio section
    audio_section = "No audio analysis available."
    if audio_result:
        pred = "Depression Detected" if audio_result['prediction'] == 1 else "No Depression"
        top_features = "N/A"
        if 'importance' in audio_result and audio_result['importance'] is not None:
            importance = audio_result['importance']
            # Get top 5 feature names
            top_idx = np.argsort(importance)[::-1][:5]
            top_names = [AUDIO_FEATURE_NAMES[i] if i < len(AUDIO_FEATURE_NAMES) else f"F{i}" for i in top_idx]
            top_features = ", ".join(top_names)
        audio_section = f"""- Prediction: {pred}
- Confidence: {audio_result['confidence']:.1%}
- Model: {audio_result.get('model_type', 'CNN')}
- Segments analyzed: {audio_result.get('segments', 'N/A')}
- Top acoustic features: {top_features}"""
    
    # Build text section
    text_section = "No text analysis available."
    if text_result:
        pred = "Severe Depression Indicators" if text_result['prediction'] == 1 else "Non-Severe / Normal"
        text_section = f"""- Prediction: {pred}
- Confidence: {text_result['confidence']:.1%}
- Model: {text_result.get('model_used', 'ML')}
- Raw text sample: \"{raw_text[:500]}{'...' if len(raw_text) > 500 else ''}\""""
    
    # Build fusion section
    fused_pred, fused_conf = fuse_predictions(
        audio_result['prediction'] if audio_result else None,
        audio_result['confidence'] if audio_result else None,
        text_result['prediction'] if text_result else None,
        text_result['confidence'] if text_result else None,
        audio_weight, text_weight
    )
    fusion_section = f"""- Method: {fusion_method}
- Audio weight: {audio_weight:.0%}, Text weight: {text_weight:.0%}
- Fused prediction: {'Depression Detected' if fused_pred == 1 else 'No Depression'}
- Fused confidence: {fused_conf:.1%}"""
    
    prompt = f"""You are a clinical AI assistant analyzing multimodal depression detection results. Review the following data and provide a concise, professional clinical-style analysis.

## AUDIO ANALYSIS
{audio_section}

## TEXT ANALYSIS
{text_section}

## MULTIMODAL FUSION
{fusion_section}

Provide your analysis covering these five areas:

1. **Tone & Speech Pattern Analysis** — What do the acoustic features (pitch, jitter, shimmer, loudness, MFCCs) reveal about the subject's vocal affect and emotional state?

2. **Linguistic Sentiment Analysis** — What do the language patterns in the text indicate about mood, cognition, and emotional well-being?

3. **Cross-Modal Consistency** — Do the audio and text results align, contradict, or complement each other? What does this tell us about the reliability of the assessment?

4. **Risk Assessment Summary** — Overall severity level, confidence in the assessment, and any red flags that stand out.

5. **Recommendations** — Suggested next steps, whether professional evaluation is warranted, and any specific areas to monitor.

Keep your response under 350 words. Use bullet points for readability. Be empathetic but clinically precise."""
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a compassionate clinical AI assistant specializing in mental health analysis."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.4,
            max_tokens=600,
            timeout=30
        )
        return response.choices[0].message.content
    except Exception as e:
        # Silently fail — don't break the app
        print(f"[LLM Analysis Error] {e}")
        return None

# ============== VISUALIZATIONS ==============
# ---- DARK THEME CHART COLORS ----
DARK_PAPER = '#161b22'
DARK_PLOT = '#0d1117'
DARK_TEXT = '#e6edf3'
DARK_GRID = '#21262d'
DARK_MUTED = '#8b949e'

def apply_dark_theme(fig, title=None):
    """Apply dark theme to any Plotly figure"""
    fig.update_layout(
        paper_bgcolor=DARK_PAPER,
        plot_bgcolor=DARK_PLOT,
        font_color=DARK_TEXT,
        title_font_color=DARK_TEXT,
        legend_font_color=DARK_MUTED,
    )
    fig.update_xaxes(
        gridcolor=DARK_GRID, linecolor=DARK_GRID, tickfont_color=DARK_MUTED,
        title_font_color=DARK_MUTED
    )
    fig.update_yaxes(
        gridcolor=DARK_GRID, linecolor=DARK_GRID, tickfont_color=DARK_MUTED,
        title_font_color=DARK_MUTED
    )
    if title:
        fig.update_layout(title=title)
    return fig

def create_modality_comparison_chart(audio_conf, text_conf, fused_conf):
    """Create comparison chart for modalities (dark theme)"""
    fig = go.Figure()
    
    categories = ['Audio', 'Text', 'Fused']
    values = [audio_conf * 100 if audio_conf else 0, 
              text_conf * 100 if text_conf else 0, 
              fused_conf * 100 if fused_conf else 0]
    colors = ['#667eea', '#3fb950', '#f85149']
    
    fig.add_trace(go.Bar(
        x=categories,
        y=values,
        marker_color=colors,
        text=[f"{v:.1f}%" for v in values],
        textposition='outside',
        textfont_color=DARK_TEXT
    ))
    
    fig.add_hline(y=50, line_dash="dash", line_color="#f85149", 
                  annotation_text="Threshold", annotation_font_color=DARK_MUTED)
    
    fig.update_layout(
        title="Detection Confidence by Modality",
        yaxis_title="Confidence %",
        yaxis_range=[0, 100],
        height=350,
    )
    apply_dark_theme(fig)
    return fig

def create_feature_importance_chart(importance, names, title):
    """Create feature importance bar chart (dark theme)"""
    sorted_idx = np.argsort(importance)[::-1][:15]
    
    fig = go.Figure(go.Bar(
        x=importance[sorted_idx],
        y=[names[i] if i < len(names) else f"F{i}" for i in sorted_idx],
        orientation='h',
        marker=dict(
            color=importance[sorted_idx],
            colorscale='Plasma'
        )
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Importance",
        height=400,
        yaxis=dict(autorange="reversed"),
    )
    apply_dark_theme(fig)
    return fig

def create_word_cloud_data(word_importance):
    """Create word importance visualization (dark theme)"""
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
    )
    apply_dark_theme(fig)
    return fig

def show_landing_page():
    """MiSec premium SaaS landing page — matches reference design."""
    # Hide sidebar on landing
    st.markdown("""
    <style>
        [data-testid="stSidebar"] {display: none !important;}
        [data-testid="collapsedControl"] {display: none !important;}
        .main .block-container {padding-left: 2rem; padding-right: 2rem; max-width: 1200px;}
    </style>
    """, unsafe_allow_html=True)

    # ========== NAVBAR ==========
    st.markdown("""
    <div class="nav-bar">
        <div class="nav-logo">🧠 MiSec</div>
        <div class="nav-links">
            <a href="#product">Product</a>
            <a href="#how-it-works">How it Works</a>
            <a href="#features">Features</a>
            <a href="#performance">Performance</a>
            <a href="#security">Security</a>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ========== HERO ==========
    hero_left, hero_right = st.columns([1.1, 1])
    with hero_left:
        st.markdown("""
        <div style="padding: 40px 0;">
            <div class="eyebrow">AI-Powered Mental Health Intelligence</div>
            <h1 class="h1-misec" style="margin-bottom:20px;">Detect emotional distress before it becomes critical.</h1>
            <p class="lead-misec" style="margin-bottom:28px;">MiSec uses multimodal AI to analyze speech, language, and behavioral patterns for early signs of depression and emotional burnout. Built for clinicians. Designed for real impact.</p>
        </div>
        """, unsafe_allow_html=True)
        c1, c2, c3 = st.columns([1.2, 1, 2])
        with c1:
            if st.button("🚀 Start Analysis →", key="hero_cta", use_container_width=True, type="primary"):
                st.session_state.page = 'app'
                st.rerun()
        with c2:
            st.button("▶ Watch Demo", key="watch_demo", use_container_width=True, type="secondary")
        st.markdown("""
        <div style="display:flex; gap:24px; margin-top:28px; flex-wrap:wrap;">
            <span style="font-size:13px; color:var(--l-muted); display:flex; align-items:center; gap:6px;">✅ HIPAA Ready</span>
            <span style="font-size:13px; color:var(--l-muted); display:flex; align-items:center; gap:6px;">⚡ Real-Time Detection</span>
            <span style="font-size:13px; color:var(--l-muted); display:flex; align-items:center; gap:6px;">🔍 Explainable AI</span>
            <span style="font-size:13px; color:var(--l-muted); display:flex; align-items:center; gap:6px;">🧪 Clinical Research Driven</span>
        </div>
        """, unsafe_allow_html=True)

    with hero_right:
        # Live Analysis preview card
        st.markdown("""
        <div style="background:var(--l-surface); border:1px solid var(--l-border); border-radius:var(--radius-lg); padding:24px; box-shadow:0 8px 32px rgba(26,22,37,0.08);">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:16px;">
                <span style="font-size:14px; font-weight:600; color:var(--l-fg);">Live Analysis</span>
                <span class="badge-pill badge-green" style="font-size:11px;">● Recording</span>
            </div>
            <div style="height:48px; background:linear-gradient(90deg, var(--grad-start), var(--grad-end)); border-radius:8px; opacity:0.15; margin-bottom:16px; position:relative; overflow:hidden;">
                <div style="position:absolute; inset:0; background:url('data:image/svg+xml,<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"200\" height=\"48\"><path d=\"M0,24 Q10,10 20,24 T40,24 T60,24 T80,24 T100,24 T120,24 T140,24 T160,24 T180,24 T200,24\" stroke=\"%23667eea\" fill=\"none\" stroke-width=\"2\"/></svg>'); opacity:0.6;"></div>
            </div>
            <div style="display:grid; grid-template-columns:1fr 1fr; gap:12px; margin-bottom:16px;">
                <div style="background:#f8f7fa; border-radius:var(--radius); padding:12px; text-align:center;">
                    <div style="font-size:11px; color:var(--l-muted); margin-bottom:4px;">Depression Risk</div>
                    <div style="font-size:24px; font-weight:700; color:var(--d-danger);">72%</div>
                    <div style="font-size:11px; color:var(--d-danger);">High Risk</div>
                </div>
                <div style="background:#f8f7fa; border-radius:var(--radius); padding:12px; text-align:center;">
                    <div style="font-size:11px; color:var(--l-muted); margin-bottom:4px;">Emotional Trend</div>
                    <div style="font-size:11px; color:var(--l-muted);">Past 7 Days</div>
                    <div class="sparkline" style="height:28px; margin-top:4px;">
                        <div class="spark-bar" style="height:40%;"></div>
                        <div class="spark-bar" style="height:60%;"></div>
                        <div class="spark-bar" style="height:35%;"></div>
                        <div class="spark-bar" style="height:70%;"></div>
                        <div class="spark-bar" style="height:50%;"></div>
                        <div class="spark-bar" style="height:80%;"></div>
                        <div class="spark-bar" style="height:65%;"></div>
                    </div>
                </div>
            </div>
            <div style="font-size:12px; color:var(--l-muted); margin-bottom:8px;">Key Indicators</div>
            <div style="display:flex; flex-wrap:wrap; gap:6px;">
                <span class="badge-pill badge-red">Low Energy</span>
                <span class="badge-pill badge-amber">Hopelessness</span>
                <span class="badge-pill badge-purple">Social Withdrawal</span>
                <span class="badge-pill badge-blue">Fatigue</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ========== TRUST LOGOS ==========
    st.markdown("""
    <div style="text-align:center; padding: 32px 0; border-top:1px solid var(--l-border); border-bottom:1px solid var(--l-border); margin: 32px 0;">
        <p style="font-size:12px; color:var(--l-muted); margin-bottom:16px; letter-spacing:0.05em; text-transform:uppercase;">Trusted by leading institutions</p>
        <div style="display:flex; justify-content:center; gap:40px; flex-wrap:wrap; align-items:center; opacity:0.5;">
            <span style="font-size:16px; font-weight:600; color:var(--l-fg); font-family:'DM Serif Display',serif;">Mayo Clinic</span>
            <span style="font-size:16px; font-weight:600; color:var(--l-fg); font-family:'DM Serif Display',serif;">Johns Hopkins</span>
            <span style="font-size:16px; font-weight:600; color:var(--l-fg); font-family:'DM Serif Display',serif;">Stanford Medicine</span>
            <span style="font-size:16px; font-weight:600; color:var(--l-fg); font-family:'DM Serif Display',serif;">UCLA Health</span>
            <span style="font-size:16px; font-weight:600; color:var(--l-fg);">NVIDIA</span>
            <span style="font-size:16px; font-weight:600; color:var(--l-fg);">Microsoft for Startups</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ========== PRODUCT PREVIEW (DARK) ==========
    st.markdown('<a id="product"></a>', unsafe_allow_html=True)
    st.markdown("""
    <div class="dark-section">
        <div style="display:grid; grid-template-columns: 1fr 2fr; gap: 48px; align-items:start;">
            <div>
                <p class="eyebrow" style="color:var(--d-accent);">Product Preview</p>
                <h2 class="h2-misec" style="color:var(--d-fg) !important; margin-bottom:16px;">Real insights.<br>Real time.</h2>
                <p class="lead-misec" style="color:var(--d-muted); margin-bottom:24px;">Our dashboard provides clinicians and teams with actionable insights powered by multimodal AI.</p>
                <div style="display:flex; flex-direction:column; gap:12px;">
                    <div style="display:flex; align-items:center; gap:10px; color:var(--d-fg); font-size:14px;"><span style="color:var(--d-success);">✓</span> Live risk scoring and confidence</div>
                    <div style="display:flex; align-items:center; gap:10px; color:var(--d-fg); font-size:14px;"><span style="color:var(--d-success);">✓</span> Speech & text analysis in real-time</div>
                    <div style="display:flex; align-items:center; gap:10px; color:var(--d-fg); font-size:14px;"><span style="color:var(--d-success);">✓</span> Emotional trend monitoring</div>
                    <div style="display:flex; align-items:center; gap:10px; color:var(--d-fg); font-size:14px;"><span style="color:var(--d-success);">✓</span> Explainable AI with highlighted cues</div>
                </div>
            </div>
            <div style="background:var(--d-surface); border:1px solid var(--d-border); border-radius:var(--radius-lg); padding:24px;">
                <div style="display:flex; gap:8px; margin-bottom:20px;">
                    <div style="flex:1; background:var(--d-surface-2); border-radius:var(--radius); padding:16px; text-align:center;">
                        <div style="font-size:11px; color:var(--d-muted); margin-bottom:4px;">Current Risk</div>
                        <div style="font-size:28px; font-weight:700; color:var(--d-danger);">72%</div>
                        <div style="font-size:11px; color:var(--d-danger);">High Risk</div>
                    </div>
                    <div style="flex:1; background:var(--d-surface-2); border-radius:var(--radius); padding:16px; text-align:center;">
                        <div style="font-size:11px; color:var(--d-muted); margin-bottom:4px;">Confidence</div>
                        <div style="font-size:28px; font-weight:700; color:var(--d-success);">86%</div>
                        <div style="font-size:11px; color:var(--d-muted);">High</div>
                    </div>
                    <div style="flex:1; background:var(--d-surface-2); border-radius:var(--radius); padding:16px; text-align:center;">
                        <div style="font-size:11px; color:var(--d-muted); margin-bottom:4px;">Analysis Length</div>
                        <div style="font-size:28px; font-weight:700; color:var(--d-fg);">3m 12s</div>
                        <div style="font-size:11px; color:var(--d-muted);">Audio</div>
                    </div>
                    <div style="flex:1; background:var(--d-surface-2); border-radius:var(--radius); padding:16px; text-align:center;">
                        <div style="font-size:11px; color:var(--d-muted); margin-bottom:4px;">Detected Signals</div>
                        <div style="font-size:28px; font-weight:700; color:var(--d-accent);">18</div>
                        <div style="font-size:11px; color:var(--d-muted);">Key Indicators</div>
                    </div>
                </div>
                <div style="display:grid; grid-template-columns: 2fr 1fr; gap:16px;">
                    <div style="background:var(--d-surface-2); border-radius:var(--radius); padding:16px;">
                        <div style="font-size:12px; color:var(--d-muted); margin-bottom:8px;">Emotional Trend</div>
                        <div class="sparkline" style="height:60px; gap:4px;">
                            <div class="spark-bar" style="height:30%; opacity:0.4;"></div>
                            <div class="spark-bar" style="height:45%; opacity:0.5;"></div>
                            <div class="spark-bar" style="height:35%; opacity:0.4;"></div>
                            <div class="spark-bar" style="height:55%; opacity:0.6;"></div>
                            <div class="spark-bar" style="height:40%; opacity:0.5;"></div>
                            <div class="spark-bar" style="height:65%; opacity:0.7;"></div>
                            <div class="spark-bar" style="height:50%; opacity:0.5;"></div>
                            <div class="spark-bar" style="height:70%; opacity:0.8;"></div>
                            <div class="spark-bar" style="height:60%; opacity:0.6;"></div>
                            <div class="spark-bar" style="height:75%; opacity:0.9;"></div>
                            <div class="spark-bar" style="height:65%; opacity:0.7;"></div>
                            <div class="spark-bar" style="height:80%; opacity:1;"></div>
                        </div>
                    </div>
                    <div style="background:var(--d-surface-2); border-radius:var(--radius); padding:16px;">
                        <div style="font-size:12px; color:var(--d-muted); margin-bottom:8px;">Top Detected Signals</div>
                        <div style="display:flex; flex-direction:column; gap:8px;">
                            <div><div style="display:flex; justify-content:space-between; font-size:11px; color:var(--d-fg); margin-bottom:2px;"><span>Hopelessness</span><span>0.82</span></div><div style="height:4px; background:var(--d-border); border-radius:2px;"><div style="width:82%; height:100%; background:var(--d-danger); border-radius:2px;"></div></div></div>
                            <div><div style="display:flex; justify-content:space-between; font-size:11px; color:var(--d-fg); margin-bottom:2px;"><span>Low Energy</span><span>0.76</span></div><div style="height:4px; background:var(--d-border); border-radius:2px;"><div style="width:76%; height:100%; background:var(--d-warn); border-radius:2px;"></div></div></div>
                            <div><div style="display:flex; justify-content:space-between; font-size:11px; color:var(--d-fg); margin-bottom:2px;"><span>Anhedonia</span><span>0.61</span></div><div style="height:4px; background:var(--d-border); border-radius:2px;"><div style="width:61%; height:100%; background:var(--d-accent); border-radius:2px;"></div></div></div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ========== HOW IT WORKS ==========
    st.markdown('<a id="how-it-works"></a>', unsafe_allow_html=True)
    st.markdown('<p class="eyebrow" style="text-align:center; margin-top:48px;">How It Works</p>', unsafe_allow_html=True)
    st.markdown('<h2 class="h2-misec" style="text-align:center; margin-bottom:40px;">Three simple steps to deeper understanding</h2>', unsafe_allow_html=True)
    s1, s2, s3 = st.columns(3)
    with s1:
        st.markdown("""
        <div class="card-misec step-card">
            <div class="step-num">01</div>
            <div class="step-title">🎤 Input</div>
            <div class="step-desc">Upload or record speech or text from conversations, sessions, or interviews.</div>
        </div>
        """, unsafe_allow_html=True)
    with s2:
        st.markdown("""
        <div class="card-misec step-card">
            <div class="step-num">02</div>
            <div class="step-title">🧠 AI Analysis</div>
            <div class="step-desc">Our multimodal AI extracts acoustic features, linguistic patterns, and emotional signals.</div>
        </div>
        """, unsafe_allow_html=True)
    with s3:
        st.markdown("""
        <div class="card-misec step-card">
            <div class="step-num">03</div>
            <div class="step-title">📊 Actionable Insights</div>
            <div class="step-desc">Get risk scores, emotional trends, and explainable insights to guide early intervention.</div>
        </div>
        """, unsafe_allow_html=True)

    # ========== FEATURES ==========
    st.markdown('<a id="features"></a>', unsafe_allow_html=True)
    st.markdown('<p class="eyebrow" style="text-align:center; margin-top:48px;">Powerful Features</p>', unsafe_allow_html=True)
    st.markdown('<h2 class="h2-misec" style="text-align:center; margin-bottom:40px;">Everything you need for early detection</h2>', unsafe_allow_html=True)
    f1, f2, f3 = st.columns(3)
    with f1:
        st.markdown("""
        <div class="card-misec">
            <div class="feature-mark">⚡</div>
            <h3>Real-Time Analysis</h3>
            <p style="color:var(--l-muted); font-size:14px; line-height:1.6;">Instantly analyze speech and text for early signs of distress.</p>
        </div>
        """, unsafe_allow_html=True)
    with f2:
        st.markdown("""
        <div class="card-misec">
            <div class="feature-mark">🔀</div>
            <h3>Multimodal AI</h3>
            <p style="color:var(--l-muted); font-size:14px; line-height:1.6;">Combines audio, NLP, and behavioral cues for higher accuracy.</p>
        </div>
        """, unsafe_allow_html=True)
    with f3:
        st.markdown("""
        <div class="card-misec">
            <div class="feature-mark">🔍</div>
            <h3>Explainable Results</h3>
            <p style="color:var(--l-muted); font-size:14px; line-height:1.6;">See exactly which words or patterns influenced the prediction.</p>
        </div>
        """, unsafe_allow_html=True)
    f4, f5, f6 = st.columns(3)
    with f4:
        st.markdown("""
        <div class="card-misec">
            <div class="feature-mark">📈</div>
            <h3>Trend Monitoring</h3>
            <p style="color:var(--l-muted); font-size:14px; line-height:1.6;">Track emotional changes over time with visual analytics.</p>
        </div>
        """, unsafe_allow_html=True)
    with f5:
        st.markdown("""
        <div class="card-misec">
            <div class="feature-mark">👥</div>
            <h3>Team Collaboration</h3>
            <p style="color:var(--l-muted); font-size:14px; line-height:1.6;">Secure dashboards and reports for care teams and clinicians.</p>
        </div>
        """, unsafe_allow_html=True)
    with f6:
        st.markdown("""
        <div class="card-misec">
            <div class="feature-mark">🔌</div>
            <h3>API & Integrations</h3>
            <p style="color:var(--l-muted); font-size:14px; line-height:1.6;">Seamlessly integrate with your existing systems and workflows.</p>
        </div>
        """, unsafe_allow_html=True)

    # ========== AI ENGINE (DARK) ==========
    st.markdown("""
    <div class="dark-section" style="margin-top:48px;">
        <p class="eyebrow" style="color:var(--d-accent);">Our AI Engine</p>
        <h2 class="h2-misec" style="color:var(--d-fg) !important; margin-bottom:32px;">Advanced multimodal architecture</h2>
        <div style="display:flex; justify-content:space-between; align-items:flex-start; gap:16px; flex-wrap:wrap;">
            <div style="text-align:center; flex:1; min-width:120px;">
                <div style="width:56px; height:56px; border-radius:50%; background:var(--d-surface-2); border:1px solid var(--d-border); display:grid; place-items:center; margin:0 auto 12px; font-size:24px;">🎤</div>
                <div style="font-weight:600; color:var(--d-fg); font-size:14px;">Audio Input</div>
                <div style="font-size:12px; color:var(--d-muted);">Speech Signal</div>
            </div>
            <div style="display:flex; align-items:center; color:var(--d-muted); font-size:20px; padding-top:16px;">→</div>
            <div style="text-align:center; flex:1; min-width:120px;">
                <div style="width:56px; height:56px; border-radius:50%; background:var(--d-surface-2); border:1px solid var(--d-border); display:grid; place-items:center; margin:0 auto 12px; font-size:24px;">📊</div>
                <div style="font-weight:600; color:var(--d-fg); font-size:14px;">Acoustic Features</div>
                <div style="font-size:12px; color:var(--d-muted);">MFCC, Pitch, Jitter</div>
            </div>
            <div style="display:flex; align-items:center; color:var(--d-muted); font-size:20px; padding-top:16px;">→</div>
            <div style="text-align:center; flex:1; min-width:120px;">
                <div style="width:56px; height:56px; border-radius:50%; background:var(--d-surface-2); border:1px solid var(--d-border); display:grid; place-items:center; margin:0 auto 12px; font-size:24px;">📝</div>
                <div style="font-weight:600; color:var(--d-fg); font-size:14px;">Text Processing</div>
                <div style="font-size:12px; color:var(--d-muted);">NLP & Semantic Understanding</div>
            </div>
            <div style="display:flex; align-items:center; color:var(--d-muted); font-size:20px; padding-top:16px;">→</div>
            <div style="text-align:center; flex:1; min-width:120px;">
                <div style="width:56px; height:56px; border-radius:50%; background:var(--d-surface-2); border:1px solid var(--d-border); display:grid; place-items:center; margin:0 auto 12px; font-size:24px;">🔀</div>
                <div style="font-weight:600; color:var(--d-fg); font-size:14px;">Multimodal Fusion</div>
                <div style="font-size:12px; color:var(--d-muted);">Cross-Modal Attention</div>
            </div>
            <div style="display:flex; align-items:center; color:var(--d-muted); font-size:20px; padding-top:16px;">→</div>
            <div style="text-align:center; flex:1; min-width:120px;">
                <div style="width:56px; height:56px; border-radius:50%; background:var(--d-surface-2); border:1px solid var(--d-border); display:grid; place-items:center; margin:0 auto 12px; font-size:24px;">🎯</div>
                <div style="font-weight:600; color:var(--d-fg); font-size:14px;">Risk Prediction</div>
                <div style="font-size:12px; color:var(--d-muted);">Depression Likelihood</div>
            </div>
            <div style="display:flex; align-items:center; color:var(--d-muted); font-size:20px; padding-top:16px;">→</div>
            <div style="text-align:center; flex:1; min-width:120px;">
                <div style="width:56px; height:56px; border-radius:50%; background:var(--d-surface-2); border:1px solid var(--d-border); display:grid; place-items:center; margin:0 auto 12px; font-size:24px;">💡</div>
                <div style="font-weight:600; color:var(--d-fg); font-size:14px;">Explainability</div>
                <div style="font-size:12px; color:var(--d-muted);">Highlighting Key Signals</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ========== PERFORMANCE ==========
    st.markdown('<a id="performance"></a>', unsafe_allow_html=True)
    st.markdown('<p class="eyebrow" style="text-align:center; margin-top:48px;">Proven Performance</p>', unsafe_allow_html=True)
    st.markdown('<h2 class="h2-misec" style="text-align:center; margin-bottom:40px;">Built on rigorous benchmarks</h2>', unsafe_allow_html=True)
    p1, p2, p3, p4, p5, p6 = st.columns(6)
    perf_stats = [
        ("92.4%", "F1-Score", "Depression Detection"),
        ("< 1.2s", "Average Inference", "Time"),
        ("50K+", "Conversations", "Analyzed"),
        ("18+", "Acoustic & Linguistic", "Biomarkers"),
        ("24/7", "Real-Time", "Monitoring"),
        ("99.9%", "Uptime &", "Reliability"),
    ]
    for col, (num, label, detail) in zip([p1, p2, p3, p4, p5, p6], perf_stats):
        with col:
            st.markdown(f"""
            <div class="perf-card">
                <div class="perf-num">{num}</div>
                <div class="perf-label">{label}</div>
                <div class="perf-detail">{detail}</div>
            </div>
            """, unsafe_allow_html=True)

    # ========== USE CASES ==========
    st.markdown('<h2 class="h2-misec" style="text-align:center; margin:48px 0 32px;">Built for Impact</h2>', unsafe_allow_html=True)
    u1, u2, u3 = st.columns(3)
    use_cases = [
        ("🏥", "Mental Health Clinics", "Enhance assessment accuracy and patient outcomes."),
        ("🖥️", "Telehealth Platforms", "Integrate AI insights into virtual care workflows."),
        ("🎓", "Universities", "Support student mental health and early intervention."),
        ("💼", "Employee Wellness", "Monitor well-being and promote healthier workplaces."),
        ("📞", "Crisis Hotlines", "Assist counselors with real-time emotional insights."),
        ("🔬", "Research Institutions", "Advance mental health research with high-quality data."),
    ]
    for col, cases in zip([u1, u2, u3], [use_cases[:2], use_cases[2:4], use_cases[4:6]]):
        with col:
            for icon, title, desc in cases:
                st.markdown(f"""
                <div class="card-misec" style="margin-bottom:16px;">
                    <div style="font-size:24px; margin-bottom:8px;">{icon}</div>
                    <h3 style="font-size:16px; margin-bottom:6px;">{title}</h3>
                    <p style="color:var(--l-muted); font-size:14px; line-height:1.5; margin:0;">{desc}</p>
                </div>
                """, unsafe_allow_html=True)

    # ========== SECURITY ==========
    st.markdown('<a id="security"></a>', unsafe_allow_html=True)
    st.markdown("""
    <div style="background:var(--l-surface); border:1px solid var(--l-border); border-radius:var(--radius-lg); padding:40px; margin-top:48px;">
        <p class="eyebrow">Security & Compliance</p>
        <h2 class="h2-misec" style="margin-bottom:8px;">Privacy-first by design</h2>
        <p class="lead-misec" style="margin-bottom:24px;">We follow the highest standards to ensure data security, privacy, and ethical AI.</p>
        <div style="display:flex; justify-content:space-between; gap:16px; flex-wrap:wrap;">
            <div style="text-align:center; flex:1; min-width:140px;">
                <div style="font-size:28px; margin-bottom:8px;">🛡️</div>
                <div style="font-weight:600; font-size:14px; color:var(--l-fg);">HIPAA Ready</div>
                <div style="font-size:12px; color:var(--l-muted);">Built to comply with HIPAA regulations.</div>
            </div>
            <div style="text-align:center; flex:1; min-width:140px;">
                <div style="font-size:28px; margin-bottom:8px;">🔒</div>
                <div style="font-weight:600; font-size:14px; color:var(--l-fg);">End-to-End Encryption</div>
                <div style="font-size:12px; color:var(--l-muted);">Data encrypted in transit and at rest.</div>
            </div>
            <div style="text-align:center; flex:1; min-width:140px;">
                <div style="font-size:28px; margin-bottom:8px;">🗑️</div>
                <div style="font-weight:600; font-size:14px; color:var(--l-fg);">No Data Retention</div>
                <div style="font-size:12px; color:var(--l-muted);">Audio is not stored permanently.</div>
            </div>
            <div style="text-align:center; flex:1; min-width:140px;">
                <div style="font-size:28px; margin-bottom:8px;">☁️</div>
                <div style="font-weight:600; font-size:14px; color:var(--l-fg);">Secure Infrastructure</div>
                <div style="font-size:12px; color:var(--l-muted);">Hosted on certified cloud infrastructure.</div>
            </div>
            <div style="text-align:center; flex:1; min-width:140px;">
                <div style="font-size:28px; margin-bottom:8px;">⚖️</div>
                <div style="font-weight:600; font-size:14px; color:var(--l-fg);">Ethical AI</div>
                <div style="font-size:12px; color:var(--l-muted);">Fair, transparent, and bias-aware models.</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ========== TESTIMONIALS ==========
    st.markdown('<h2 class="h2-misec" style="text-align:center; margin:48px 0 32px;">What clinicians are saying</h2>', unsafe_allow_html=True)
    t1, t2, t3 = st.columns(3)
    testimonials = [
        ("AF", "Dr. Amanda Fields", "Clinical Psychologist", "MiSec has become an essential part of our early screening process. The insights are incredibly accurate and actionable."),
        ("MH", "Dr. Marcus Hill", "Director of Mental Health", "The explainable AI gives us confidence in our results and helps us better communicate with our patients."),
        ("PN", "Dr. Priya Nair", "Telehealth Specialist", "A game-changer for telehealth. We can now identify at-risk patients earlier and provide timely support."),
    ]
    for col, (initial, name, role, quote) in zip([t1, t2, t3], testimonials):
        with col:
            st.markdown(f"""
            <div class="testi-card">
                <div class="testi-quote">"{quote}"</div>
                <div class="testi-author">
                    <div class="testi-avatar">{initial}</div>
                    <div>
                        <div class="testi-name">{name}</div>
                        <div class="testi-role">{role}</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ========== CTA FOOTER ==========
    st.markdown("""
    <div class="cta-section" style="margin-top:48px;">
        <h2 class="h2-misec">AI that listens when humans miss the signs.</h2>
        <p class="lead-misec">Early detection. Better outcomes. Healthier lives.</p>
        <div style="display:flex; gap:16px; justify-content:center; flex-wrap:wrap; position:relative; z-index:1;">
            <a class="btn-primary-misec btn-white" href="#" onclick="return false;">🚀 Try Live Demo →</a>
            <a class="btn-secondary-misec" href="#" onclick="return false;" style="color:white; border-color:rgba(255,255,255,0.4);">Request Enterprise Access</a>
        </div>
    </div>
    """, unsafe_allow_html=True)
    # Functional CTA buttons
    cta_cols = st.columns([1, 1, 1])
    with cta_cols[1]:
        if st.button("🚀 Start Analysis →", key="footer_cta", use_container_width=True, type="primary"):
            st.session_state.page = 'app'
            st.rerun()

    # ========== PAGE FOOTER ==========
    st.markdown("""
    <div class="pagefoot">
        <p><strong>© 2024 MiSec · Multimodal Depression Detection</strong></p>
        <p class="meta-misec">Built with care for early mental health awareness. Not a substitute for professional diagnosis.</p>
    </div>
    """, unsafe_allow_html=True)


# ============== MAIN APP ==============
if 'page' not in st.session_state:
    st.session_state.page = 'landing'

if st.session_state.page == 'landing':
    show_landing_page()
    st.stop()

# ============================================================
# DARK DASHBOARD APP PAGE
# ============================================================

# Inject dark theme overrides for Streamlit components
st.markdown("""
<style>
    .stApp { background: #0d1117 !important; }
    [data-testid="stAppViewContainer"] { background: #0d1117; }
    .main .block-container { background: #0d1117; }
    h1, h2, h3, h4, p, li, span, label, .stMarkdown { color: #e6edf3 !important; }
    [data-testid="stSidebar"] {
        background: #161b22 !important;
        border-right: 1px solid #30363d !important;
    }
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] li,
    [data-testid="stSidebar"] span, [data-testid="stSidebar"] .stMarkdown,
    [data-testid="stSidebar"] strong, [data-testid="stSidebar"] label {
        color: #e6edf3 !important;
    }
    [data-testid="stSidebar"] [data-testid="stMetricValue"] { color: #e6edf3 !important; }
    [data-testid="stSidebar"] [data-testid="stMetricLabel"] { color: #8b949e !important; }
    .stTabs [data-baseweb="tab"] {
        color: #8b949e !important; background: transparent !important;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea, #764ba2) !important;
        color: white !important;
    }
        [data-testid="stMarkdownContainer"] h1,
    [data-testid="stMarkdownContainer"] h2,
    [data-testid="stMarkdownContainer"] h3,
    [data-testid="stMarkdownContainer"] h4 {
        color: #e6edf3 !important;
    }
    .stButton > button[kind="secondary"] {
        background: #21262d !important; color: #e6edf3 !important;
        border-color: #30363d !important;
    }
    .stButton > button[kind="secondary"]:hover {
        border-color: #a371f7 !important; background: #161b22 !important;
    }
    textarea, input, [data-baseweb="textarea"] textarea,
    [data-baseweb="input"] input {
        background: #161b22 !important; color: #e6edf3 !important;
        border-color: #30363d !important;
    }
    [data-testid="stSelectbox"] > div[data-baseweb="select"] {
        background: #161b22 !important; color: #e6edf3 !important;
        border-color: #30363d !important;
    }
    [data-testid="stSlider"] [data-testid="stThumbValue"] { color: #e6edf3 !important; }
    [data-testid="stRadio"] label { color: #e6edf3 !important; }
    [data-testid="stFileUploader"] > section {
        border-color: #30363d !important; background: #161b22 !important;
    }
    [data-testid="stFileUploader"] span { color: #e6edf3 !important; }
    [data-testid="stMetricValue"] { color: #e6edf3 !important; }
    [data-testid="stMetricLabel"] { color: #8b949e !important; }
    hr { border-color: #30363d !important; }
    .stSuccess, .stInfo, .stWarning {
        background: #161b22 !important; border-color: #30363d !important;
    }
    .stSuccess p, .stInfo p, .stWarning p { color: #e6edf3 !important; }
</style>
""", unsafe_allow_html=True)

col_h1, col_h2 = st.columns([5, 1])
with col_h1:
    st.markdown('<h1 style="font-family:DM Serif Display,serif; font-size:28px; color:#e6edf3 !important; margin:0;">🧠 MiSec Dashboard</h1>', unsafe_allow_html=True)
    st.markdown('<p style="color:#8b949e; font-size:13px; margin:4px 0 0 0;">Multimodal Depression Detection · Audio + Text + XAI</p>', unsafe_allow_html=True)
with col_h2:
    st.write("")
    if st.button("← Home", key="back_to_landing"):
        st.session_state.page = 'landing'
        st.rerun()

st.markdown("<hr style='border-color:#30363d; margin:16px 0;'>", unsafe_allow_html=True)

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

with st.sidebar:
    st.markdown('<p style="font-size:11px; color:#8b949e; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:16px;">MiSec AI Platform</p>', unsafe_allow_html=True)
    nav_items = [("📊", "Overview", True), ("📈", "Analytics", False), ("👤", "Patients", False), ("📋", "Reports", False), ("🔔", "Alerts", False), ("⚙️", "Settings", False)]
    for icon, label, active in nav_items:
        cls = "active" if active else ""
        st.markdown(f'<div class="dash-nav-item {cls}"><span>{icon}</span> {label}</div>', unsafe_allow_html=True)
    st.markdown("<hr style='border-color:#30363d; margin:24px 0;'>", unsafe_allow_html=True)
    st.markdown('<p style="font-size:11px; color:#8b949e; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:16px;">Configuration</p>', unsafe_allow_html=True)
    fusion_method = st.selectbox("Fusion Method", ["Weighted Average", "Audio Priority", "Text Priority", "Max Confidence"])
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
    st.markdown("<hr style='border-color:#30363d; margin:24px 0;'>", unsafe_allow_html=True)
    st.markdown('<p style="font-size:11px; color:#8b949e; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:16px;">Model Performance</p>', unsafe_allow_html=True)
    if audio_model_info:
        st.metric("Audio CNN Acc", f"{audio_model_info.get('file_accuracy', 0)*100:.1f}%")
        st.metric("Audio CNN F1", f"{audio_model_info.get('file_f1', 0)*100:.1f}%")
    else:
        st.metric("Audio Model", "Not trained")
    if text_model_info:
        best_name = text_model_info.get('best_model_name', 'Unknown')
        best_metrics = text_model_info.get('metrics', {}).get(best_name, {})
        if best_metrics:
            st.metric(f"Text ({best_name}) Acc", f"{best_metrics.get('Accuracy', 0)*100:.1f}%")
            st.metric(f"Text ({best_name}) F1", f"{best_metrics.get('F1 Score', 0)*100:.1f}%")
    else:
        st.metric("Text Model", "Not trained")

if 'audio_result' not in st.session_state:
    st.session_state.audio_result = None
if 'text_result' not in st.session_state:
    st.session_state.text_result = None

audio_r = st.session_state.audio_result
text_r = st.session_state.text_result

current_risk = "—"
risk_color = ""
if audio_r and text_r:
    fused_c = (audio_r['confidence'] * audio_weight + text_r['confidence'] * text_weight)
    current_risk = f"{fused_c*100:.0f}%"
    risk_color = "danger" if fused_c > 0.5 else "success"
elif audio_r:
    current_risk = f"{audio_r['confidence']*100:.0f}%"
    risk_color = "danger" if audio_r['confidence'] > 0.5 else "success"
elif text_r:
    current_risk = f"{text_r['confidence']*100:.0f}%"
    risk_color = "danger" if text_r['confidence'] > 0.5 else "success"

conf_val = "—"
if audio_r and text_r:
    conf_val = f"{max(audio_r['confidence'], text_r['confidence'])*100:.0f}%"
elif audio_r:
    conf_val = f"{audio_r['confidence']*100:.0f}%"
elif text_r:
    conf_val = f"{text_r['confidence']*100:.0f}%"

m1, m2, m3, m4 = st.columns(4)
with m1:
    sub = "High Risk" if risk_color=="danger" else ("Low Risk" if risk_color=="success" else "Awaiting analysis")
    st.markdown(f"""
    <div class="dash-metric-card">
        <div class="dash-metric-label">Current Risk</div>
        <div class="dash-metric-value {risk_color}">{current_risk}</div>
        <div class="dash-metric-sub">{sub}</div>
    </div>
    """, unsafe_allow_html=True)
with m2:
    st.markdown(f"""
    <div class="dash-metric-card">
        <div class="dash-metric-label">Confidence</div>
        <div class="dash-metric-value">{conf_val}</div>
        <div class="dash-metric-sub">Highest modality score</div>
    </div>
    """, unsafe_allow_html=True)
with m3:
    segments = audio_r['segments'] if audio_r else 0
    st.markdown(f"""
    <div class="dash-metric-card">
        <div class="dash-metric-label">Segments Analyzed</div>
        <div class="dash-metric-value">{segments}</div>
        <div class="dash-metric-sub">Audio chunks processed</div>
    </div>
    """, unsafe_allow_html=True)
with m4:
    signals = 0
    if audio_r: signals += 8
    if text_r: signals += 10
    st.markdown(f"""
    <div class="dash-metric-card">
        <div class="dash-metric-label">Detected Signals</div>
        <div class="dash-metric-value" style="color:#a371f7;">{signals}</div>
        <div class="dash-metric-sub">Key indicators found</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["🎤 Audio Analysis", "📝 Text Analysis", "🔀 Multimodal Fusion"])

with tab1:
    st.markdown("<h3 style='color:#e6edf3 !important; margin-bottom:4px;'>🎤 Audio-Based Depression Detection</h3>", unsafe_allow_html=True)
    st.markdown("<p style='color:#8b949e; font-size:13px; margin-bottom:16px;'>Using Improved 3-Layer CNN with BatchNorm</p>", unsafe_allow_html=True)
    if not OPENSMILE_AVAILABLE:
        st.warning("⚠️ OpenSMILE is not installed. Audio analysis is disabled. Text analysis still works fully.")

    # Audio source selector
    audio_source = st.radio(
        "Choose audio source",
        ["📁 Upload File", "🎙️ Record Live"],
        horizontal=True,
        label_visibility="collapsed"
    )

    audio_obj = None   # Original UploadedFile / BytesIO for playback (keeps MIME type)
    audio_bytes = None # Raw bytes for analysis pipeline
    audio_label = ""

    if audio_source == "📁 Upload File":
        uploaded_audio = st.file_uploader("Upload a WAV audio file", type=['wav'], key='audio_upload')
        if uploaded_audio:
            audio_obj = uploaded_audio
            audio_bytes = uploaded_audio.getvalue()
            audio_label = uploaded_audio.name
    else:
        st.markdown("""
        <p style='color:#8b949e; font-size:13px; margin-bottom:8px;'>
            Click the microphone to start recording. Speak clearly for at least 10–20 seconds for best results.<br>
            <span style='font-size:12px; opacity:0.7;'>💡 If recording is silent, check your browser mic permissions or try Chrome/Safari. Brave's privacy shields may block audio capture.</span>
        </p>
        """, unsafe_allow_html=True)
        recorded_audio = st.audio_input("Record your voice", key='audio_record')
        if recorded_audio:
            audio_obj = recorded_audio
            audio_bytes = recorded_audio.getvalue() if hasattr(recorded_audio, 'getvalue') else recorded_audio
            audio_label = "recorded_audio.wav"

    # Shared analysis pipeline for both upload and recorded audio
    if audio_obj and audio_bytes:
        c1, c2 = st.columns([2, 1])
        with c1:
            # Pass the original file-like object to st.audio so it uses the correct MIME type
            st.audio(audio_obj)
        with c2:
            st.metric("Source", "Recorded" if audio_source == "🎙️ Record Live" else "Upload")
            st.metric("Size", f"{len(audio_bytes) / 1024:.1f} KB")

        if st.button("🔍 Analyze Audio", key='analyze_audio', disabled=not OPENSMILE_AVAILABLE):
            with st.spinner("Processing audio with Improved CNN..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
                    tmp.write(audio_bytes)
                    tmp_path = tmp.name
                smile = load_opensmile()
                features = extract_audio_features(tmp_path, smile)
                if features is not None:
                    segments = create_audio_segments(features)
                    if segments is not None:
                        device = get_device()
                        audio_model, model_type, num_ch = load_audio_model()
                        if audio_model is None:
                            model_type = 'SimpleCNN'
                            audio_model = AudioCNNModel(segments.shape[2]).to(device)
                        predictions, probabilities = predict_audio(audio_model, segments, device)
                        importance = compute_audio_importance(audio_model, segments, device)
                        vote = Counter(predictions).most_common(1)[0][0]
                        avg_conf = np.mean(probabilities)
                        st.session_state.audio_result = {
                            'prediction': int(vote), 'confidence': avg_conf,
                            'importance': importance, 'segments': len(segments),
                            'model_type': model_type
                        }
                        st.markdown("<hr style='border-color:#30363d; margin:24px 0;'>", unsafe_allow_html=True)
                        st.markdown("<h3 style='color:#e6edf3 !important;'>📈 Audio Analysis Results</h3>", unsafe_allow_html=True)
                        r1, r2, r3, r4 = st.columns(4)
                        with r1: st.metric("Segments", len(segments))
                        with r2: st.metric("Confidence", f"{avg_conf:.1%}")
                        with r3: st.metric("Result", "Depression" if vote == 1 else "Normal")
                        with r4: st.metric("Model", model_type)
                        st.markdown("<h4 style='color:#e6edf3 !important; margin-top:16px;'>🔍 Audio Feature Importance</h4>", unsafe_allow_html=True)
                        fig = create_feature_importance_chart(importance, AUDIO_FEATURE_NAMES, "Top Audio Features (Gradient Saliency)")
                        st.plotly_chart(fig, use_container_width=True)
                os.unlink(tmp_path)

with tab2:
    st.markdown("<h3 style='color:#e6edf3 !important; margin-bottom:4px;'>📝 Text-Based Depression Detection</h3>", unsafe_allow_html=True)
    st.markdown("<p style='color:#8b949e; font-size:13px; margin-bottom:16px;'>Using Enhanced NLP with Negation Handling</p>", unsafe_allow_html=True)
    text_input = st.text_area("Enter text to analyze", height=200, placeholder="Type or paste text here...")
    model_choices = ["Best ML Model (Tuned)", "Ensemble (Soft Voting)", "Attention BiLSTM"]
    text_model_choice = st.radio("Select Model", model_choices, horizontal=True)
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
                else:
                    lstm_data = load_lstm_model()
                    if lstm_data:
                        pred, conf = predict_text_lstm(text_input, lstm_data['vocab'], lstm_data['lstm'], device, max_len=lstm_data.get('max_len', 256))
                    else:
                        st.error("LSTM model not loaded.")
                if pred is not None:
                    st.session_state.text_result = {
                        'prediction': pred, 'confidence': conf,
                        'text': text_input, 'model_used': used_model
                    }
                    st.markdown("<hr style='border-color:#30363d; margin:24px 0;'>", unsafe_allow_html=True)
                    st.markdown("<h3 style='color:#e6edf3 !important;'>📈 Text Analysis Results</h3>", unsafe_allow_html=True)
                    tr1, tr2, tr3 = st.columns(3)
                    with tr1:
                        severity = "Severe" if pred == 1 else "Non-Severe"
                        st.metric("Severity", severity)
                    with tr2: st.metric("Confidence", f"{conf:.1%}")
                    with tr3: st.metric("Model", used_model.split('(')[0].strip())
                    fi = load_feature_importance()
                    if fi:
                        word_imp = get_text_word_importance(text_input, None, fi)
                        if word_imp:
                            st.markdown("<h4 style='color:#e6edf3 !important; margin-top:16px;'>🔍 Key Words Analysis</h4>", unsafe_allow_html=True)
                            fig = create_word_cloud_data(word_imp)
                            if fig:
                                st.plotly_chart(fig, use_container_width=True)
                    st.markdown("<h4 style='color:#e6edf3 !important; margin-top:16px;'>💡 Interpretation</h4>", unsafe_allow_html=True)
                    if pred == 1:
                        st.markdown("""
                        <div style="background:#161b22; border:1px solid #f85149; border-radius:10px; padding:16px;">
                            <p style="color:#f85149; font-weight:600; margin:0 0 8px 0;">⚠️ Severe depression indicators detected</p>
                            <p style="color:#8b949e; font-size:14px; margin:0;">The text contains language patterns associated with hopelessness, self-harm ideation, and deep emotional distress.</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                        <div style="background:#161b22; border:1px solid #3fb950; border-radius:10px; padding:16px;">
                            <p style="color:#3fb950; font-weight:600; margin:0 0 8px 0;">✅ Non-severe indicators</p>
                            <p style="color:#8b949e; font-size:14px; margin:0;">The text contains mild language patterns associated with general depressive symptoms.</p>
                        </div>
                        """, unsafe_allow_html=True)
        else:
            st.warning("Please enter some text to analyze.")

with tab3:
    st.markdown("<h3 style='color:#e6edf3 !important; margin-bottom:4px;'>🔀 Multimodal Fusion Analysis</h3>", unsafe_allow_html=True)
    st.markdown("<p style='color:#8b949e; font-size:13px; margin-bottom:16px;'>Combine audio and text for comprehensive assessment</p>", unsafe_allow_html=True)
    audio_result = st.session_state.audio_result
    text_result = st.session_state.text_result
    fs1, fs2 = st.columns(2)
    with fs1:
        st.markdown("<h4 style='color:#e6edf3 !important;'>🎤 Audio Status</h4>", unsafe_allow_html=True)
        if audio_result:
            st.success(f"✅ Analyzed ({audio_result['segments']} segments)")
            st.write(f"Prediction: {'Depression' if audio_result['prediction'] == 1 else 'Normal'}")
            st.write(f"Confidence: {audio_result['confidence']:.1%}")
        else:
            st.info("⏳ No audio analyzed yet")
    with fs2:
        st.markdown("<h4 style='color:#e6edf3 !important;'>📝 Text Status</h4>", unsafe_allow_html=True)
        if text_result:
            st.success(f"✅ Analyzed — {text_result.get('model_used', 'ML')}")
            st.write(f"Prediction: {'Severe' if text_result['prediction'] == 1 else 'Non-Severe'}")
            st.write(f"Confidence: {text_result['confidence']:.1%}")
        else:
            st.info("⏳ No text analyzed yet")
    st.markdown("<hr style='border-color:#30363d; margin:24px 0;'>", unsafe_allow_html=True)
    if audio_result or text_result:
        st.markdown("<h3 style='color:#e6edf3 !important;'>🎯 Fused Prediction</h3>", unsafe_allow_html=True)
        audio_pred = audio_result['prediction'] if audio_result else None
        audio_conf = audio_result['confidence'] if audio_result else None
        text_pred = text_result['prediction'] if text_result else None
        text_conf = text_result['confidence'] if text_result else None
        if fusion_method == "Max Confidence":
            if audio_conf and text_conf:
                fused_pred, fused_conf = (audio_pred, audio_conf) if audio_conf > text_conf else (text_pred, text_conf)
            else:
                fused_pred, fused_conf = fuse_predictions(audio_pred, audio_conf, text_pred, text_conf)
        else:
            fused_pred, fused_conf = fuse_predictions(audio_pred, audio_conf, text_pred, text_conf, audio_weight, text_weight)
        if fused_pred is not None:
            fp1, fp2 = st.columns([1, 1])
            with fp1:
                if fused_pred == 1:
                    st.markdown("""
                    <div class="pred-card-dark positive">
                        <h2 style="color:#f85149 !important; margin-bottom:8px;">⚠️ Depression Detected</h2>
                        <p style="color:#8b949e; margin:0;">Multimodal analysis indicates signs of depression</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="pred-card-dark negative">
                        <h2 style="color:#3fb950 !important; margin-bottom:8px;">✅ No Depression Detected</h2>
                        <p style="color:#8b949e; margin:0;">Multimodal analysis shows normal patterns</p>
                    </div>
                    """, unsafe_allow_html=True)
            with fp2:
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=fused_conf * 100,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Fused Confidence", 'font': {'color': '#e6edf3'}},
                    number={'font': {'color': '#e6edf3'}},
                    gauge={
                        'axis': {'range': [0, 100], 'tickfont': {'color': '#8b949e'}},
                        'bar': {'color': "#667eea"},
                        'bgcolor': '#161b22', 'bordercolor': '#30363d',
                        'steps': [
                            {'range': [0, 50], 'color': "#21262d"},
                            {'range': [50, 75], 'color': "#d29922"},
                            {'range': [75, 100], 'color': "#f85149"}
                        ],
                        'threshold': {
                            'line': {'color': "#f85149", 'width': 4},
                            'thickness': 0.75, 'value': 50
                        }
                    }
                ))
                fig.update_layout(height=250, paper_bgcolor='#0d1117', font_color='#e6edf3')
                st.plotly_chart(fig, use_container_width=True)
            st.markdown("<h3 style='color:#e6edf3 !important; margin-top:24px;'>📊 Modality Comparison</h3>", unsafe_allow_html=True)
            fig = create_modality_comparison_chart(audio_conf, text_conf, fused_conf)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("<h3 style='color:#e6edf3 !important; margin-top:24px;'>💡 Multimodal Explanation</h3>", unsafe_allow_html=True)
            explanation_parts = []
            if audio_result:
                audio_text = "detected depression indicators in speech patterns"
                if audio_result['prediction'] == 0:
                    audio_text = "found normal speech patterns"
                explanation_parts.append(f"**Audio** ({audio_result.get('model_type', 'CNN')}): {audio_text}")
            if text_result:
                text_text = "identified severe depression language patterns"
                if text_result['prediction'] == 0:
                    text_text = "found mild depression indicators"
                explanation_parts.append(f"**Text** ({text_result.get('model_used', 'ML')}): {text_text}")
            explanation_parts.append(f"**Fusion ({fusion_method})**: Audio {audio_weight:.0%} · Text {text_weight:.0%}")
            for part in explanation_parts:
                st.markdown(f"<p style='color:#8b949e; font-size:14px;'>{part}</p>", unsafe_allow_html=True)
            
            # ---- LLM DEEP ANALYSIS (backend, minimal UI) ----
            if audio_result and text_result:
                # Build a hash of current inputs to know if we need to regenerate
                current_hash = f"{audio_result['prediction']}-{audio_result['confidence']:.4f}-{text_result['prediction']}-{text_result['confidence']:.4f}-{fusion_method}"
                prev_hash = st.session_state.get('llm_input_hash', '')
                
                if current_hash != prev_hash:
                    st.session_state.llm_input_hash = current_hash
                    st.session_state.llm_analysis = None  # clear old analysis
                
                if st.session_state.get('llm_analysis') is None:
                    with st.spinner("🤖 Generating AI-powered deep analysis..."):
                        raw_text = text_result.get('text', '')
                        llm_result = generate_llm_analysis(
                            audio_result, text_result,
                            fusion_method, audio_weight, text_weight,
                            raw_text=raw_text
                        )
                        st.session_state.llm_analysis = llm_result
                
                if st.session_state.get('llm_analysis'):
                    with st.expander("🤖 AI-Powered Deep Analysis", expanded=False):
                        st.markdown(f"""
                        <div style="background:#161b22; border:1px solid #30363d; border-radius:10px; padding:20px;">
                            <p style="color:#e6edf3; font-size:14px; line-height:1.7;">{st.session_state.llm_analysis.replace(chr(10), '<br>')}</p>
                        </div>
                        """, unsafe_allow_html=True)
    else:
        st.info("👆 Please analyze audio and/or text in the other tabs first.")

st.markdown("<hr style='border-color:#30363d; margin:32px 0 16px 0;'>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center; padding:1rem 0;">
    <p style="color:#8b949e; font-size:13px; margin:0 0 8px 0;">⚠️ <strong style="color:#e6edf3;">Disclaimer:</strong> For research and educational purposes only. Not a substitute for professional medical diagnosis.</p>
    <p style="color:#484f58; font-size:12px; margin:0;">Built with ❤️ using Streamlit, PyTorch, OpenSMILE & NLP | Enhanced Models v2</p>
</div>
""", unsafe_allow_html=True)
