#!/usr/bin/env python3
"""
Hero vs Villain Classification
CLIP ViT-L/14 + Feature Importance
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
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import shap
import pandas as pd

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# ============================================================
# 1. LOAD CLIP
# ============================================================
print("\nLoading CLIP ViT-L/14 (laion2b_s32b_b82k)...")

model, _, preprocess = open_clip.create_model_and_transforms(
    model_name="ViT-L-14",
    pretrained="laion2b_s32b_b82k"
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
        feat = model.encode_image(t)
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
               if x.lower().endswith(("jpg","png","jpeg"))]

villain_images = [os.path.join(villain_dir, x)
                  for x in os.listdir(villain_dir)
                  if x.lower().endswith(("jpg","png","jpeg"))]

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

# Save features for later use
np.save("clip_features.npy", X)
np.save("clip_labels.npy", y)
print("✓ Saved clip_features.npy and clip_labels.npy\n")

# ============================================================
# 5. SPLIT
# ============================================================
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.4, stratify=y, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
)

# ============================================================
# 6. CLASSIFIER (MLP)
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
# 8. TEST
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
print("\n✓ Model Finished\n")

# ============================================================
# 9. FEATURE IMPORTANCE ANALYSIS
# ============================================================
print("Running feature importance analysis...\n")

# ---------- Logistic Regression ----------
log_reg = LogisticRegression(max_iter=500)
log_reg.fit(X, y)
log_importance = log_reg.coef_[0]

# ---------- Random Forest ----------
rf = RandomForestClassifier(n_estimators=500, random_state=42)
rf.fit(X, y)
rf_importance = rf.feature_importances_

# ---------- SHAP ----------
print("Computing SHAP values (may take time)...")
background = X[:50]
explainer = shap.KernelExplainer(log_reg.predict_proba, background)
shap_values_all = explainer.shap_values(background)[1]
shap_mean = shap_values_all.mean(axis=0)

# ---------- FIX SHAP DIMENSION ----------
feature_dim = X.shape[1]
if shap_mean.shape[0] > feature_dim:
    shap_mean = shap_mean[:feature_dim]
elif shap_mean.shape[0] < feature_dim:
    pad = np.zeros(feature_dim - shap_mean.shape[0])
    shap_mean = np.concatenate([shap_mean, pad])

# ============================================================
# BUILD TABLE
# ============================================================
feature_indices = np.arange(feature_dim)
feature_names = [f"clip_dim_{i}" for i in feature_indices]

df = pd.DataFrame({
    "feature_index": feature_indices,
    "feature_name": feature_names,
    "logistic_weight": log_importance,
    "rf_importance": rf_importance,
    "shap_mean": shap_mean
})

df["logistic_rank"] = df["logistic_weight"].rank(ascending=False, method="dense").astype(int)
df["rf_rank"] = df["rf_importance"].rank(ascending=False, method="dense").astype(int)
df["shap_rank"] = df["shap_mean"].rank(ascending=False, method="dense").astype(int)

df_sorted = df.sort_values("shap_rank")
df_sorted.to_csv("feature_importance_table.csv", index=False)

np.savetxt("logistic_importance.csv", log_importance, delimiter=",")
np.savetxt("random_forest_importance.csv", rf_importance, delimiter=",")
np.savetxt("shap_values.csv", shap_mean, delimiter=",")

# ---------- Summary file ----------
top10 = df_sorted.head(10)

with open("feature_importance_summary.txt", "w") as f:
    f.write("Hero vs Villain – CLIP ViT-L/14 Feature Importance Summary\n")
    f.write("=========================================================\n\n")
    f.write(f"Total features: {feature_dim}\n")
    f.write(f"Total samples:  {X.shape[0]}\n\n")

    f.write("Top 10 features by SHAP:\n")
    for _, row in top10.iterrows():
        f.write(
            f"  - {row['feature_name']} (index {int(row['feature_index'])}): "
            f"SHAP={row['shap_mean']:.4f}, "
            f"log_w={row['logistic_weight']:.4f}, "
            f"rf_imp={row['rf_importance']:.4f}\n"
        )

print("\n✓ Feature importance analysis complete.")
print("  - feature_importance_table.csv")
print("  - logistic_importance.csv")
print("  - random_forest_importance.csv")
print("  - shap_values.csv")
print("  - feature_importance_summary.txt")