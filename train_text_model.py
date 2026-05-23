"""
Text-Based Depression Detection — Boosting Techniques
XGBoost, LightGBM, AdaBoost, Gradient Boosting, Stacking & Voting Ensembles
No GridSearch — uses optimized hyperparameters directly for speed.
"""

import pandas as pd
import numpy as np
import re
import pickle
import warnings
import os
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, classification_report, roc_auc_score
)
from sklearn.calibration import CalibratedClassifierCV

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import (
    GradientBoostingClassifier, VotingClassifier, 
    AdaBoostClassifier, StackingClassifier
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# Deep Learning
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

# NLP
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')
    nltk.download('stopwords')
    nltk.download('wordnet')

print("=" * 60)
print("DEPRESSION DETECTION — BOOSTING TECHNIQUES")
print("=" * 60)

# ============== 1. LOAD DATA ==============
print("\n[1] Loading Dataset...")
DATA_PATH = "/Users/sahilchauhan/Downloads/multimodaldetection/audiomodels/reddit_depression_suicidewatch 2.csv"
df = pd.read_csv(DATA_PATH)
print(f"   Samples: {len(df)}")
df['severity_label'] = df['label'].apply(lambda x: 1 if x == 'SuicideWatch' else 0)
print(f"   Depression: {(df['severity_label']==0).sum()}, SuicideWatch: {(df['severity_label']==1).sum()}")

# ============== 2. PREPROCESS ==============
print("\n[2] Text Preprocessing...")

lemmatizer = WordNetLemmatizer()
try:
    stop_words = set(stopwords.words('english'))
except:
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))

KEEP = {'not','no','nor','never','nothing','nobody','cannot','without','very','too','more',
        'most','only','just','should','would','could','might','myself','again','why','how',
        'all','few','own','same','than','down','out','off','over','under'}
stop_words = stop_words - KEEP
NEGATION = {'not','no','never','neither','nobody','nothing','nowhere','nor','cannot'}

CONTRACTIONS = {
    "can't":"cannot","won't":"will not","don't":"do not","doesn't":"does not",
    "didn't":"did not","isn't":"is not","aren't":"are not","wasn't":"was not",
    "weren't":"were not","haven't":"have not","hasn't":"has not","hadn't":"had not",
    "wouldn't":"would not","shouldn't":"should not","couldn't":"could not",
    "i'm":"i am","i've":"i have","i'll":"i will","i'd":"i would",
    "it's":"it is","that's":"that is","there's":"there is","they're":"they are",
    "we're":"we are","you're":"you are",
}

def preprocess_text(text):
    if pd.isna(text): return ""
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    for c, e in CONTRACTIONS.items():
        text = text.replace(c, e)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    try: tokens = word_tokenize(text)
    except: tokens = text.split()
    
    processed = []
    negate = False
    nc = 0
    for t in tokens:
        if t in NEGATION:
            negate = True; nc = 0; processed.append(t); continue
        if negate:
            if t in {'but','however','although','though'} or nc >= 3:
                negate = False; processed.append(t)
            else:
                processed.append(f'NOT_{t}'); nc += 1
        else:
            processed.append(t)
    
    final = []
    for t in processed:
        if t.startswith('NOT_'):
            base = t[4:]
            if len(base) > 2: final.append(f'NOT_{lemmatizer.lemmatize(base)}')
        elif t not in stop_words and len(t) > 2:
            final.append(lemmatizer.lemmatize(t))
    return ' '.join(final)

df['cleaned_text'] = df['text'].apply(preprocess_text)
df = df[df['cleaned_text'].str.len() > 10]
print(f"   Clean samples: {len(df)}")

# ============== 3. TF-IDF ==============
print("\n[3] TF-IDF Feature Extraction...")
X = df['cleaned_text']
y = df['severity_label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

tfidf = TfidfVectorizer(max_features=15000, ngram_range=(1,3), min_df=3, max_df=0.95, sublinear_tf=True)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)
print(f"   Train: {X_train_tfidf.shape}, Test: {X_test_tfidf.shape}")

# ============== 4. TRAIN ALL BOOSTERS ==============
print("\n" + "=" * 60)
print("[4] TRAINING BOOSTING MODELS (Direct — No GridSearch)")
print("=" * 60)

results = []

def train_eval(name, model):
    print(f"\n   Training {name}...")
    model.fit(X_train_tfidf, y_train)
    yp = model.predict(X_test_tfidf)
    acc = accuracy_score(y_test, yp)
    prec = precision_score(y_test, yp, zero_division=0)
    rec = recall_score(y_test, yp, zero_division=0)
    f1 = f1_score(y_test, yp, zero_division=0)
    try:
        if hasattr(model, 'predict_proba'):
            auc = roc_auc_score(y_test, model.predict_proba(X_test_tfidf)[:,1])
        elif hasattr(model, 'decision_function'):
            auc = roc_auc_score(y_test, model.decision_function(X_test_tfidf))
        else: auc = 0.0
    except: auc = 0.0
    results.append({'Model': name, 'Accuracy': acc, 'Precision': prec, 
                     'Recall': rec, 'F1': f1, 'AUC': auc, 'obj': model})
    print(f"      Acc={acc:.4f}  Prec={prec:.4f}  Rec={rec:.4f}  F1={f1:.4f}  AUC={auc:.4f}")
    return model

# --- XGBoost ---
xgb = train_eval('XGBoost', XGBClassifier(
    n_estimators=500, max_depth=6, learning_rate=0.1,
    subsample=0.8, colsample_bytree=0.8, min_child_weight=1,
    eval_metric='logloss', tree_method='hist', n_jobs=-1, random_state=42
))

# --- LightGBM ---
lgb = train_eval('LightGBM', LGBMClassifier(
    n_estimators=500, max_depth=-1, learning_rate=0.1,
    subsample=0.8, colsample_bytree=0.8, num_leaves=63,
    min_child_samples=10, verbose=-1, n_jobs=-1, random_state=42
))

# --- Gradient Boosting (sklearn) ---
gb = train_eval('Gradient Boosting', GradientBoostingClassifier(
    n_estimators=300, max_depth=5, learning_rate=0.1,
    subsample=0.8, random_state=42
))

# --- AdaBoost ---
ada = train_eval('AdaBoost', AdaBoostClassifier(
    n_estimators=200, learning_rate=0.1, algorithm='SAMME', random_state=42
))

# --- Logistic Regression (baseline) ---
lr = train_eval('Logistic Regression', LogisticRegression(
    C=0.5, max_iter=2000, random_state=42
))

# --- Calibrated SVM (baseline) ---
svm_cal = train_eval('Linear SVM (Cal)', CalibratedClassifierCV(
    LinearSVC(C=0.1, max_iter=3000, random_state=42, dual='auto'), cv=3
))

# --- Stacking Ensemble ---
print("\n   Training Stacking Ensemble (XGB+LGB+GB+SVM → LR meta)...")
stacking = StackingClassifier(
    estimators=[('xgb', xgb), ('lgb', lgb), ('gb', gb), ('svm', svm_cal)],
    final_estimator=LogisticRegression(max_iter=2000, random_state=42),
    cv=3, n_jobs=-1, passthrough=False
)
train_eval('Stacking Ensemble', stacking)

# --- Soft Voting ---
print("\n   Training Voting Ensemble (XGB+LGB+LR+SVM)...")
voting = VotingClassifier(
    estimators=[('xgb', xgb), ('lgb', lgb), ('lr', lr), ('svm', svm_cal)],
    voting='soft', n_jobs=-1
)
train_eval('Voting Ensemble', voting)


# ============== 5. ATTENTION BiLSTM ==============
print("\n\n[5] Attention BiLSTM...")

class TextDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len=256):
        self.texts = texts; self.labels = labels
        self.vocab = vocab; self.max_len = max_len
    def __len__(self): return len(self.texts)
    def __getitem__(self, idx):
        tokens = self.texts.iloc[idx].split()[:self.max_len]
        ids = [self.vocab.get(t, 1) for t in tokens]
        ids += [0] * (self.max_len - len(ids))
        return torch.tensor(ids), torch.tensor(self.labels.iloc[idx], dtype=torch.float32)

class Attention(nn.Module):
    def __init__(self, hd):
        super().__init__()
        self.attn = nn.Sequential(nn.Linear(hd, hd//2), nn.Tanh(), nn.Linear(hd//2, 1))
    def forward(self, x):
        w = torch.softmax(self.attn(x), dim=1)
        return torch.sum(w * x, dim=1), w

class AttentionBiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, hidden_dim=256, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True, dropout=0.3)
        self.attention = Attention(hidden_dim*2)
        self.fc1 = nn.Linear(hidden_dim*2, 128)
        self.fc2 = nn.Linear(128, 1)
        self.dropout = nn.Dropout(0.4)
        self.ln = nn.LayerNorm(hidden_dim*2)
    def forward(self, x):
        e = self.dropout(self.embedding(x))
        o, _ = self.lstm(e)
        o = self.ln(o)
        c, _ = self.attention(o)
        c = self.dropout(c)
        c = torch.relu(self.fc1(c))
        c = self.dropout(c)
        return torch.sigmoid(self.fc2(c)).squeeze()

def build_vocab(texts, max_vocab=20000):
    wc = {}
    for t in texts:
        for w in t.split(): wc[w] = wc.get(w, 0) + 1
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for w, c in sorted(wc.items(), key=lambda x: x[1], reverse=True)[:max_vocab-2]:
        if c >= 2: vocab[w] = len(vocab)
    return vocab

vocab = build_vocab(X_train, 20000)
print(f"   Vocab: {len(vocab)}")

MAX_LEN = 256
train_ds = TextDataset(X_train.reset_index(drop=True), y_train.reset_index(drop=True), vocab, MAX_LEN)
test_ds = TextDataset(X_test.reset_index(drop=True), y_test.reset_index(drop=True), vocab, MAX_LEN)
train_dl = DataLoader(train_ds, batch_size=64, shuffle=True)
test_dl = DataLoader(test_ds, batch_size=64, shuffle=False)

device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
print(f"   Device: {device}")

model = AttentionBiLSTMClassifier(len(vocab)).to(device)
criterion = nn.BCELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)

best_f1 = 0; patience = 5; pc = 0; best_state = None
print("   Training...")
for ep in range(15):
    model.train(); tloss = 0
    for tx, lb in train_dl:
        tx, lb = tx.to(device), lb.to(device)
        optimizer.zero_grad()
        out = model(tx)
        loss = criterion(out, lb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        tloss += loss.item()
    avg = tloss / len(train_dl)
    scheduler.step(avg)
    
    model.eval(); vp, vl = [], []
    with torch.no_grad():
        for tx, lb in test_dl:
            out = model(tx.to(device))
            vp.extend((out.cpu().numpy() > 0.5).astype(int))
            vl.extend(lb.numpy())
    vf1 = f1_score(vl, vp, zero_division=0)
    vacc = accuracy_score(vl, vp)
    print(f"      Ep {ep+1}/15  Loss={avg:.4f}  Acc={vacc:.4f}  F1={vf1:.4f}")
    if vf1 > best_f1:
        best_f1 = vf1; best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}; pc = 0
    else:
        pc += 1
        if pc >= patience: print(f"      Early stop ep {ep+1}"); break

if best_state: model.load_state_dict(best_state); model = model.to(device)

model.eval(); ap, al, aprob = [], [], []
with torch.no_grad():
    for tx, lb in test_dl:
        out = model(tx.to(device))
        p = out.cpu().numpy()
        ap.extend((p > 0.5).astype(int)); al.extend(lb.numpy()); aprob.extend(p)

la = accuracy_score(al, ap); lp = precision_score(al, ap, zero_division=0)
lr_val = recall_score(al, ap, zero_division=0); lf = f1_score(al, ap, zero_division=0)
lauc = roc_auc_score(al, aprob)
results.append({'Model':'Attention BiLSTM','Accuracy':la,'Precision':lp,'Recall':lr_val,'F1':lf,'AUC':lauc,'obj':model})
print(f"   BiLSTM: Acc={la:.4f}  Prec={lp:.4f}  Rec={lr_val:.4f}  F1={lf:.4f}  AUC={lauc:.4f}")


# ============== 6. RESULTS ==============
print("\n" + "=" * 60)
print("FINAL MODEL COMPARISON")
print("=" * 60)

rdf = pd.DataFrame([{k:v for k,v in r.items() if k!='obj'} for r in results]).sort_values('F1', ascending=False)
print(rdf.to_string(index=False))

best = max(results, key=lambda x: x['F1'])
best_trad = max([r for r in results if r['Model'] != 'Attention BiLSTM'], key=lambda x: x['F1'])

print(f"\n🏆 BEST OVERALL: {best['Model']} — F1={best['F1']:.4f}  Acc={best['Accuracy']:.4f}  AUC={best['AUC']:.4f}")
print(f"🏆 BEST TRADITIONAL: {best_trad['Model']} — F1={best_trad['F1']:.4f}  Acc={best_trad['Accuracy']:.4f}")

print(f"\n📋 Classification Report ({best_trad['Model']}):")
yp = best_trad['obj'].predict(X_test_tfidf)
print(classification_report(y_test, yp, target_names=['Depression','SuicideWatch'], digits=4))


# ============== 7. SAVE ==============
print("\n[7] Saving Models...")
save_dir = "/Users/sahilchauhan/Downloads/multimodaldetection/saved_models"
os.makedirs(save_dir, exist_ok=True)

with open(f"{save_dir}/tfidf_vectorizer.pkl", 'wb') as f: pickle.dump(tfidf, f)
with open(f"{save_dir}/lstm_vocab.pkl", 'wb') as f: pickle.dump(vocab, f)
with open(f"{save_dir}/best_text_model.pkl", 'wb') as f: pickle.dump(best_trad['obj'], f)

# Always save LSTM state_dict on CPU to avoid MPS device issues in app.py
cpu_state = {k: v.detach().clone().cpu() for k, v in model.state_dict().items()}
torch.save(cpu_state, f"{save_dir}/lstm_model.pth")
print("   ✅ LSTM saved on CPU (device-independent)")

for r in results:
    if r['Model'] in ['XGBoost','LightGBM','Stacking Ensemble','Voting Ensemble']:
        nm = r['Model'].lower().replace(' ','_')
        with open(f"{save_dir}/{nm}_model.pkl", 'wb') as f: pickle.dump(r['obj'], f)

# Also save as ensemble_model.pkl (voting ensemble) for app.py compatibility
voting_result = next((r for r in results if r['Model'] == 'Voting Ensemble'), None)
if voting_result:
    with open(f"{save_dir}/ensemble_model.pkl", 'wb') as f: pickle.dump(voting_result['obj'], f)
    print("   ✅ Saved ensemble_model.pkl (Voting Ensemble)")

model_info = {
    'best_model_name': best['Model'],
    'best_traditional_model': best_trad['Model'],
    'metrics': {r['Model']: {k:v for k,v in r.items() if k!='obj'} for r in results},
    'vocab_size': len(vocab), 'tfidf_features': X_train_tfidf.shape[1],
    'max_len': MAX_LEN, 'lstm_type': 'AttentionBiLSTM',
}
with open(f"{save_dir}/model_info.pkl", 'wb') as f: pickle.dump(model_info, f)

# Feature importance — use LR coefficients (works for word-level XAI in app.py)
try:
    fnames = tfidf.get_feature_names_out().tolist()
    # LR coefficients: positive = SuicideWatch, negative = Depression
    lr_coefs = lr.coef_[0].tolist()
    # XGB importance as bonus
    xgb_imp = xgb.feature_importances_.tolist()
    top_idx = sorted(range(len(xgb_imp)), key=lambda i: xgb_imp[i], reverse=True)[:20]
    print("\n   Top 20 XGBoost Features:")
    for i in top_idx: print(f"      {fnames[i]}: {xgb_imp[i]:.4f}")
    fi = {
        'feature_names': fnames,
        'coefficients': lr_coefs,     # ← key app.py expects
        'xgb_importance': xgb_imp,
    }
    with open(f"{save_dir}/feature_importance.pkl", 'wb') as f: pickle.dump(fi, f)
    print("   ✅ Saved feature_importance.pkl with 'coefficients' key")
except Exception as e:
    print(f"   ⚠️ Feature importance save failed: {e}")

print(f"\n   Saved all models to: {save_dir}")

print("\n" + "=" * 60)
print("✅ TRAINING COMPLETE!")
print("=" * 60)
print(f"🏆 Best: {best['Model']} — F1={best['F1']:.4f}  Acc={best['Accuracy']:.4f}  AUC={best['AUC']:.4f}")
