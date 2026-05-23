import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import pandas as pd
import os
import copy
from collections import Counter

# Ensure determinism in results
torch.manual_seed(0)
np.random.seed(0)

# Check for GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {device}")

# Define directory paths for Mac
base_dir = "/Users/sahilchauhan/Downloads/multimodaldetection"
directory = os.path.join(base_dir, "Audio-Interview")
fold_csv_path = os.path.join(base_dir, "foldlist/InterviewFolds.csv")

# Read fold information
fold_info = pd.read_csv(fold_csv_path)
fold_files = [fold_info[fold].dropna().tolist() for fold in fold_info.columns]

# Get list of available feature files
available_files = [f.replace('_it_audio_features.csv', '') for f in os.listdir(directory) if f.endswith('_audio_features.csv')]
print(f"Available feature files: {len(available_files)}")
print(f"Files: {available_files}")


def load_features_for_fold(directory, filenames):
    segments = []
    labels = []
    identifiers = []
    epsilon = 1e-8

    for identifier in filenames:
        filename = f"{identifier}_it_audio_features.csv"
        path = os.path.join(directory, filename)
        if os.path.exists(path):
            df = pd.read_csv(path)
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            features = df[numeric_cols].values

            label = 1 if 'PM' in identifier or 'PF' in identifier else 0
            L, step = 128, 64

            for start in range(0, len(features) - L + 1, step):
                segment = features[start:start + L]
                std_dev = np.std(segment, axis=0)
                zero_variance = std_dev == 0
                std_dev[zero_variance] = 1
                normalized_segment = (segment - np.mean(segment, axis=0)) / (std_dev + epsilon)
                segments.append(normalized_segment)
                labels.append(label)
                identifiers.append(identifier)
        else:
            print(f"Feature file not found: {path}")

    if len(segments) == 0:
        return np.array([]), np.array([]), []
    
    return np.array(segments), np.array(labels), identifiers


# ============== IMPROVED CNN MODEL ==============
class ImprovedCNNModel(nn.Module):
    """Deeper CNN with BatchNorm, multiple conv layers, and global average pooling"""
    def __init__(self, num_channels):
        super(ImprovedCNNModel, self).__init__()
        
        # Conv Block 1
        self.conv1 = nn.Conv1d(num_channels, 64, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Conv Block 2
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Conv Block 3
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        
        # Global Average Pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Classification head
        self.fc1 = nn.Linear(256, 64)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, x):
        # Conv Block 1
        x = self.pool1(self.relu(self.bn1(self.conv1(x))))
        
        # Conv Block 2
        x = self.pool2(self.relu(self.bn2(self.conv2(x))))
        
        # Conv Block 3
        x = self.relu(self.bn3(self.conv3(x)))
        
        # Global average pooling
        x = self.global_pool(x)  # (batch, 256, 1)
        x = x.squeeze(-1)       # (batch, 256)
        
        # Classification
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return self.sigmoid(x)


# Keep the simple model for compatibility/fallback
class SimpleCNNModel(nn.Module):
    def __init__(self, num_channels):
        super(SimpleCNNModel, self).__init__()
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
        x = self.sigmoid(x)
        return x


def majority_voting(predictions, identifiers):
    """Aggregate predictions at the file level."""
    votes = {}
    for identifier, prediction in zip(identifiers, predictions):
        if identifier not in votes:
            votes[identifier] = []
        votes[identifier].append(prediction)
    final_predictions = {identifier: Counter(preds).most_common(1)[0][0] for identifier, preds in votes.items()}
    return final_predictions


def extract_true_labels(identifiers):
    """Return the true label for each file/identifier."""
    return {identifier: 1 if 'PM' in identifier or 'PF' in identifier else 0 for identifier in set(identifiers)}


def augment_data(X, y, noise_factor=0.005, num_augmentations=2):
    """Enhanced data augmentation with noise injection, time shifting, and scaling"""
    augmented_X = [X]
    augmented_y = [y]
    
    for _ in range(num_augmentations):
        # Gaussian noise
        noisy = X + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=X.shape)
        augmented_X.append(noisy)
        augmented_y.append(y)
    
    # Time masking: randomly zero out a time range
    masked = X.copy()
    for i in range(len(masked)):
        mask_len = np.random.randint(5, 20)
        mask_start = np.random.randint(0, max(1, masked.shape[2] - mask_len))
        masked[i, :, mask_start:mask_start + mask_len] = 0
    augmented_X.append(masked)
    augmented_y.append(y)
    
    # Feature scaling: slight random scale per sample
    scaled = X * (1.0 + 0.1 * np.random.randn(X.shape[0], 1, 1))
    augmented_X.append(scaled)
    augmented_y.append(y)
    
    return np.concatenate(augmented_X, axis=0), np.concatenate(augmented_y, axis=0)


# ======== TRAINING ========
print("\n" + "="*60)
print("IMPROVED CNN: Running with available extracted features")
print("="*60)

# Collect all available data
all_segments = []
all_labels = []
all_identifiers = []

for identifier in available_files:
    filename = f"{identifier}_it_audio_features.csv"
    path = os.path.join(directory, filename)
    
    if os.path.exists(path):
        df = pd.read_csv(path)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        features = df[numeric_cols].values
        
        label = 1 if 'PM' in identifier or 'PF' in identifier else 0
        L, step = 128, 64
        epsilon = 1e-8
        
        for start in range(0, len(features) - L + 1, step):
            segment = features[start:start + L]
            std_dev = np.std(segment, axis=0)
            zero_variance = std_dev == 0
            std_dev[zero_variance] = 1
            normalized_segment = (segment - np.mean(segment, axis=0)) / (std_dev + epsilon)
            
            all_segments.append(normalized_segment)
            all_labels.append(label)
            all_identifiers.append(identifier)

if len(all_segments) == 0:
    print("ERROR: No segments could be extracted. Please run feature extraction first.")
    exit(1)

print(f"Total segments: {len(all_segments)}")
print(f"Unique files: {len(set(all_identifiers))}")
print(f"Label distribution: {Counter(all_labels)}")

# Convert to numpy arrays
X_all = np.array(all_segments)
y_all = np.array(all_labels)

# Split 80/20 for training/testing
from sklearn.model_selection import train_test_split
indices = list(range(len(X_all)))
train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42, stratify=y_all)

X_train = X_all[train_idx]
y_train = y_all[train_idx]
X_test = X_all[test_idx]
y_test = y_all[test_idx]
test_identifiers = [all_identifiers[i] for i in test_idx]

print(f"\nTraining samples (before augmentation): {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

# Transpose for Conv1D: (batch, features, sequence_length)
X_train = X_train.transpose(0, 2, 1)
X_test = X_test.transpose(0, 2, 1)

# Enhanced data augmentation
X_train_aug, y_train_aug = augment_data(X_train, y_train, noise_factor=0.005, num_augmentations=2)
print(f"Training samples (after augmentation): {len(X_train_aug)}")

# Shuffle augmented data
shuffle_idx = np.random.permutation(len(X_train_aug))
X_train_aug = X_train_aug[shuffle_idx]
y_train_aug = y_train_aug[shuffle_idx]

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train_aug, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train_aug, dtype=torch.float32).unsqueeze(1).to(device)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)

train_data = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_data, shuffle=True, batch_size=32)

# Get number of channels from the data
num_channels = X_train.shape[1]
print(f"Number of feature channels: {num_channels}")

# ===== Train Improved CNN =====
print("\n--- Training Improved CNN (3-layer with BatchNorm) ---")
model = ImprovedCNNModel(num_channels=num_channels).to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4)
scheduler = CosineAnnealingLR(optimizer, T_max=50)

best_f1 = 0
best_model_state = None
num_epochs = 50

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        epoch_loss += loss.item()
    
    scheduler.step()
    
    # Evaluate every 5 epochs
    if (epoch + 1) % 5 == 0:
        model.eval()
        with torch.no_grad():
            outputs = model(X_test_tensor)
            preds = (outputs.squeeze() > 0.5).float().cpu().numpy()
        
        val_f1 = f1_score(y_test, preds, zero_division=0)
        val_acc = accuracy_score(y_test, preds)
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss/len(train_loader):.4f}, "
              f"Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")
        
        # Save best model
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_model_state = copy.deepcopy(model.state_dict())

# Restore best model
if best_model_state is not None:
    model.load_state_dict(best_model_state)

print(f"\nBest validation F1: {best_f1:.4f}")

# Final evaluation
print("\nEvaluating improved model...")
model.eval()
with torch.no_grad():
    outputs = model(X_test_tensor)
    probabilities = outputs.squeeze().cpu().numpy()
    predictions = (probabilities > 0.5).astype(float)

# Segment-level metrics
print("\n" + "="*60)
print("SEGMENT-LEVEL RESULTS (Improved CNN)")
print("="*60)
print(classification_report(y_test, predictions, target_names=['Not Depressed (C)', 'Depressed (P)'], digits=4))

# File-level metrics (majority voting)
voted_predictions = majority_voting(predictions, test_identifiers)
true_labels = extract_true_labels(test_identifiers)

unique_test_files = list(set(test_identifiers))
y_true_file = [true_labels[f] for f in unique_test_files]
y_pred_file = [voted_predictions.get(f, 0) for f in unique_test_files]

print("\n" + "="*60)
print("FILE-LEVEL RESULTS (Majority Voting)")
print("="*60)
for f in unique_test_files:
    print(f"  {f}: True={true_labels[f]}, Pred={voted_predictions.get(f, 'N/A')}")

file_acc = accuracy_score(y_true_file, y_pred_file)
file_prec = precision_score(y_true_file, y_pred_file, zero_division=0)
file_rec = recall_score(y_true_file, y_pred_file, zero_division=0)
file_f1 = f1_score(y_true_file, y_pred_file, zero_division=0)

print(f"\nFile-level Accuracy: {file_acc:.4f}")
print(f"File-level Precision: {file_prec:.4f}")
print(f"File-level Recall: {file_rec:.4f}")
print(f"File-level F1 Score: {file_f1:.4f}")

# Save improved model
save_dir = os.path.join(base_dir, "saved_models")
os.makedirs(save_dir, exist_ok=True)
torch.save(model.state_dict(), os.path.join(save_dir, "improved_audio_cnn.pth"))

# Also save model info
import pickle
audio_model_info = {
    'model_type': 'ImprovedCNN',
    'num_channels': num_channels,
    'segment_accuracy': accuracy_score(y_test, predictions),
    'segment_f1': f1_score(y_test, predictions, zero_division=0),
    'file_accuracy': file_acc,
    'file_f1': file_f1,
    'file_precision': file_prec,
    'file_recall': file_rec,
}
with open(os.path.join(save_dir, "audio_model_info.pkl"), 'wb') as f:
    pickle.dump(audio_model_info, f)

print(f"\nSaved improved audio model to: {save_dir}/improved_audio_cnn.pth")

print("\n" + "="*60)
print("Model training and evaluation complete!")
print("="*60)
