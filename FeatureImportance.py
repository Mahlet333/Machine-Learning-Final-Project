#!/usr/bin/env python3
"""
Hero vs Villain Classification
CLIP ViT-L/14 (LAION2B S32B B82K)
Fully deep-learning, no handpicked features
"""

# ============================================================
# IMPORTS
# ============================================================
import os
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)
import torch.nn as nn
import torch.optim as optim
import open_clip

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# ============================================================
# 1. LOAD CLIP ViT-L/14 (LAION2B_S32B_B82K)
# ============================================================
print("\nLoading CLIP ViT-L/14 (laion2b_s32b_b82k)...")

model, _, preprocess = open_clip.create_model_and_transforms(
    model_name="ViT-L-14",
    pretrained="laion2b_s32b_b82k"     # THIS ONE EXISTS ON YOUR SYSTEM
)

model = model.to(device)
model.eval()

print("✓ CLIP Loaded\n")

# ============================================================
# 2. FEATURE EXTRACTION
# ============================================================
def extract_features(path):
    try:
        img = Image.open(path).convert("RGB")
    except:
        return None

    with torch.no_grad():
        t = preprocess(img).unsqueeze(0).to(device)
        feat = model.encode_image(t)    # ~768 dims
        feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat.cpu().numpy().flatten()

# ============================================================
# 3. LOAD DATA
# ============================================================
print("Loading dataset...")
hero_dir = "mugshots_google_Hero"
villain_dir = "mugshots_google_Anti"

hero_images = [os.path.join(hero_dir, x)
               for x in os.listdir(hero_dir)
               if x.lower().endswith(("jpg", "png", "jpeg"))]

villain_images = [os.path.join(villain_dir, x)
                  for x in os.listdir(villain_dir)
                  if x.lower().endswith(("jpg", "png", "jpeg"))]

all_images = hero_images + villain_images
labels = [0]*len(hero_images) + [1]*len(villain_images)

print(f"Total images: {len(all_images)}")

# ============================================================
# 4. EXTRACT FEATURES
# ============================================================
print("\nExtracting CLIP features...")
X, y = [], []

for img_path, label in tqdm(zip(all_images, labels), total=len(all_images)):
    f = extract_features(img_path)
    if f is None:
        continue
    X.append(f)
    y.append(label)

X = np.array(X)
y = np.array(y)

print("Final feature shape:", X.shape)

# ============================================================
# 5. TRAIN/VAL/TEST SPLIT
# ============================================================
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.4, stratify=y, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
)

# ============================================================
# 6. CLASSIFIER
# ============================================================
class MLP(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        return self.layers(x)

model_clf = MLP(X.shape[1]).to(device)
optimizer = optim.Adam(model_clf.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

def t(x):
    return torch.tensor(x, dtype=torch.float32, device=device)

# ============================================================
# 7. TRAINING
# ============================================================
print("\nTraining classifier...")
best_val = 0

for epoch in range(25):
    model_clf.train()
    optimizer.zero_grad()

    out = model_clf(t(X_train))
    loss = criterion(out, torch.tensor(y_train, dtype=torch.long, device=device))
    loss.backward()
    optimizer.step()

    model_clf.eval()
    with torch.no_grad():
        val_out = model_clf(t(X_val))
        _, pred = torch.max(val_out, 1)
        val_acc = (pred.cpu().numpy() == y_val).mean()

    print(f"Epoch {epoch+1}/25 | Loss={loss.item():.4f} | Val Acc={val_acc:.4f}")

    if val_acc > best_val:
        best_val = val_acc
        torch.save(model_clf.state_dict(), "best_clip_classifier.pth")

print("Best validation accuracy:", best_val)

# ============================================================
# 8. TEST EVALUATION
# ============================================================
model_clf.load_state_dict(torch.load("best_clip_classifier.pth"))

with torch.no_grad():
    out = model_clf(t(X_test))
    probs = torch.softmax(out, dim=1)[:, 1].cpu().numpy()
    _, pred = torch.max(out, 1)
    pred = pred.cpu().numpy()

acc = accuracy_score(y_test, pred)
prec = precision_score(y_test, pred)
rec = recall_score(y_test, pred)
f1 = f1_score(y_test, pred)
auc = roc_auc_score(y_test, probs)
cm = confusion_matrix(y_test, pred)

print("\nTEST RESULTS:")
print("Accuracy:", acc)
print("Precision:", prec)
print("Recall:", rec)
print("F1:", f1)
print("AUC:", auc)
print("Confusion Matrix:\n", cm)

print("\n✓ Done. CLIP ViT-L/14 pipeline working.")