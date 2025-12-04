#!/usr/bin/env python3
"""
Hero vs Villain Classification
CLIP ViT-L/14 + Feature Importance + Graphs
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
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import RocCurveDisplay

# Get the project root directory (2 levels up from this script)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models", "checkpoints")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(os.path.join(RESULTS_DIR, "features"), exist_ok=True)
os.makedirs(os.path.join(RESULTS_DIR, "visualizations"), exist_ok=True)

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
hero_dir = os.path.join(PROJECT_ROOT, "data", "images", "hero")
villain_dir = os.path.join(PROJECT_ROOT, "data", "images", "villain")

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

np.save(os.path.join(RESULTS_DIR, "features", "clip_features.npy"), X)
np.save(os.path.join(RESULTS_DIR, "features", "clip_labels.npy"), y)
print("✓ Saved clip_features.npy and clip_labels.npy\n")

# ============================================================
# 5. SPLIT DATA
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

def t(x): return torch.tensor(x, dtype=torch.float32, device=device)

# ============================================================
# 7. TRAIN MODEL
# ============================================================
train_losses = []
val_accuracies = []

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
    train_losses.append(loss.item())
    val_accuracies.append(val_acc)

    if val_acc > best_val:
        best_val = val_acc
        torch.save(model_clf.state_dict(), os.path.join(MODELS_DIR, "best_clip_classifier.pth"))

print("\nBest validation accuracy:", best_val)

# ============================================================
# 8. TEST EVALUATION
# ============================================================
model_clf.load_state_dict(torch.load(os.path.join(MODELS_DIR, "best_clip_classifier.pth")))
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
print("\n✓ Test evaluation complete\n")

# ============================================================
# 9. TRAINING & TEST GRAPHS
# ============================================================
# ---- Training Loss Curve
plt.figure(figsize=(8,5))
plt.plot(train_losses, label="Training Loss", color="blue")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Curve")
plt.grid(True)
plt.legend()
plt.savefig(os.path.join(RESULTS_DIR, "visualizations", "training_loss_curve.png"), dpi=300)
plt.show()

# ---- Validation Accuracy Curve
plt.figure(figsize=(8,5))
plt.plot(val_accuracies, label="Validation Accuracy", color="green")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Validation Accuracy Curve")
plt.grid(True)
plt.legend()
plt.savefig(os.path.join(RESULTS_DIR, "visualizations", "validation_accuracy_curve.png"), dpi=300)
plt.show()

# ---- Confusion Matrix
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Hero", "Villain"],
            yticklabels=["Hero", "Villain"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.savefig(os.path.join(RESULTS_DIR, "visualizations", "confusion_matrix.png"), dpi=300)
plt.show()

# ---- ROC Curve
plt.figure(figsize=(7,6))
RocCurveDisplay.from_predictions(y_test, probs)
plt.plot([0,1],[0,1],'k--')
plt.title(f"ROC Curve (AUC = {auc:.3f})")
plt.grid(True)
plt.savefig(os.path.join(RESULTS_DIR, "visualizations", "roc_curve.png"), dpi=300)
plt.show()

# ============================================================
# 10. FEATURE IMPORTANCE ANALYSIS
# ============================================================
print("Running feature importance analysis...\n")

# ---- Logistic Regression ----
log_reg = LogisticRegression(max_iter=500)
log_reg.fit(X, y)
log_importance = log_reg.coef_[0]

# ---- Random Forest ----
rf = RandomForestClassifier(n_estimators=500, random_state=42)
rf.fit(X, y)
rf_importance = rf.feature_importances_

# ---- SHAP ----
print("Computing SHAP values (may take time)...")
background = X[:50]
explainer = shap.KernelExplainer(log_reg.predict_proba, background)
shap_values_all = explainer.shap_values(background)[1]
shap_mean = shap_values_all.mean(axis=0)

# Fix SHAP mismatch
if shap_mean.shape[0] < X.shape[1]:
    pad = np.zeros(X.shape[1] - shap_mean.shape[0])
    shap_mean = np.concatenate([shap_mean, pad])

# ============================================================
# PLOT Feature Importance
# ============================================================
# ---- Top Logistic Weights ----
top_idx = np.argsort(np.abs(log_importance))[-20:]
plt.figure(figsize=(10,6))
plt.barh(range(len(top_idx)), log_importance[top_idx], color="purple")
plt.yticks(range(len(top_idx)), [f"dim_{i}" for i in top_idx])
plt.xlabel("Weight")
plt.title("Top 20 Logistic Regression Feature Weights")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "visualizations", "logistic_weights_top20.png"), dpi=300)
plt.show()

# ---- Top Random Forest Importances ----
top_idx = np.argsort(rf_importance)[-20:]
plt.figure(figsize=(10,6))
plt.barh(range(len(top_idx)), rf_importance[top_idx], color="orange")
plt.yticks(range(len(top_idx)), [f"dim_{i}" for i in top_idx])
plt.xlabel("Importance")
plt.title("Top 20 Random Forest Feature Importances")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "visualizations", "rf_importances_top20.png"), dpi=300)
plt.show()

# ---- SHAP Summary Plot ----
try:
    shap.summary_plot(shap_values_all, background, show=False)
    plt.savefig(os.path.join(RESULTS_DIR, "visualizations", "shap_summary_plot.png"), dpi=300)
    plt.show()
except:
    print("SHAP summary plot could not be generated.")

# ---- Save CSV files ----
df = pd.DataFrame({
    "feature_index": np.arange(X.shape[1]),
    "logistic_weight": log_importance,
    "rf_importance": rf_importance,
    "shap_mean": shap_mean
})
df_sorted = df.sort_values("shap_mean", ascending=False)
df_sorted.to_csv(os.path.join(RESULTS_DIR, "features", "feature_importance_table.csv"), index=False)

np.savetxt(os.path.join(RESULTS_DIR, "features", "logistic_importance.csv"), log_importance, delimiter=",")
np.savetxt(os.path.join(RESULTS_DIR, "features", "random_forest_importance.csv"), rf_importance, delimiter=",")
np.savetxt(os.path.join(RESULTS_DIR, "features", "shap_values.csv"), shap_mean, delimiter=",")

print("\n✓ Feature importance analysis complete.")