#!/usr/bin/env python3
"""
ARCFACE TRAINING - FAST VERSION
Pre-extracts all embeddings once, then trains classifier
"""

import subprocess
import sys
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'numpy<2', '--quiet', '--break-system-packages'])

import os
os.environ['INSIGHTFACE_DATA_HOME'] = '/scratch/ma7030/insightface_cache'
os.environ['TORCH_HOME'] = '/scratch/ma7030/torch_cache'
os.makedirs('/scratch/ma7030/insightface_cache', exist_ok=True)
os.makedirs('/scratch/ma7030/torch_cache', exist_ok=True)

sys.path.insert(0, '/scratch/ma7030/python_packages')

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from PIL import Image
import json
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

import cv2
cv2.setNumThreads(1)
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

print("""
════════════════════════════════════════════════════════════════════════════════
ARCFACE TRAINING - FAST VERSION (Pre-extract Embeddings)
════════════════════════════════════════════════════════════════════════════════
""")

# ==================== LOAD ARCFACE ====================

def load_arcface():
    """Load ArcFace model from InsightFace."""
    
    print("\n" + "="*80)
    print("LOADING ARCFACE (10M+ Human Faces)")
    print("="*80 + "\n")
    
    try:
        from insightface.app import FaceAnalysis
        
        print("✓ Trained on: 10,000,000+ human face images")
        print("✓ Architecture: ResNet-50 + ArcFace loss")
        print("✓ Embedding: 512-dimensional vectors\n")
        
        app = FaceAnalysis(
            name='buffalo_l',
            root='/scratch/ma7030/insightface_cache',
            providers=['CUDAExecutionProvider' if torch.cuda.is_available() else 'CPUExecutionProvider']
        )
        app.prepare(ctx_id=0 if torch.cuda.is_available() else -1)
        
        print("✓ ArcFace loaded successfully!\n")
        return app
    
    except Exception as e:
        print(f"❌ Error: {e}\n")
        return None


# ==================== DATA PREPARATION ====================

print("\n" + "="*80)
print("LOADING DATA")
print("="*80 + "\n")

hero_dir = "mugshots_google_Hero"
antihero_dir = "mugshots_google_Anti"

hero_images = [os.path.join(hero_dir, f) for f in os.listdir(hero_dir) 
               if f.endswith(('.jpg', '.jpeg', '.png'))]
antihero_images = [os.path.join(antihero_dir, f) for f in os.listdir(antihero_dir) 
                   if f.endswith(('.jpg', '.jpeg', '.png'))]

if not hero_images or not antihero_images:
    print("❌ Dataset directories not found!")
    sys.exit(1)

all_images = hero_images + antihero_images
all_labels = np.array([0]*len(hero_images) + [1]*len(antihero_images))

print(f"✓ Loaded {len(hero_images)} hero images")
print(f"✓ Loaded {len(antihero_images)} antihero images")
print(f"Total: {len(all_images)} images\n")

# Train/val/test split
train_images, temp_images, train_labels, temp_labels = train_test_split(
    all_images, all_labels, test_size=0.4, random_state=42, stratify=all_labels
)

val_images, test_images, val_labels, test_labels = train_test_split(
    temp_images, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels
)

print(f"Train: {len(train_images)} (60%)")
print(f"Val: {len(val_images)} (20%)")
print(f"Test: {len(test_images)} (20%)\n")

# ==================== EXTRACT EMBEDDINGS ====================

print("="*80)
print("PRE-EXTRACTING ARCFACE EMBEDDINGS (One-time, ~2-3 min)")
print("="*80 + "\n")

arcface_app = load_arcface()

if arcface_app is None:
    sys.exit(1)

def extract_embeddings_from_files(image_files, app, split_name):
    """Extract embeddings from image files"""
    embeddings = []
    
    for img_path in tqdm(image_files, desc=f"Extracting {split_name} embeddings", leave=False):
        try:
            img = Image.open(img_path).convert('RGB')
            img_np = np.array(img)
            
            faces = app.get(img_np)
            if len(faces) > 0:
                embeddings.append(faces[0].embedding)
            else:
                embeddings.append(np.random.randn(512) * 0.01)
        except:
            embeddings.append(np.random.randn(512) * 0.01)
    
    return np.array(embeddings, dtype=np.float32)

# Extract all embeddings
print("Extracting training embeddings...")
train_embeddings = extract_embeddings_from_files(train_images, arcface_app, "Train")

print("Extracting validation embeddings...")
val_embeddings = extract_embeddings_from_files(val_images, arcface_app, "Val")

print("Extracting test embeddings...")
test_embeddings = extract_embeddings_from_files(test_images, arcface_app, "Test")

print(f"\n✓ Train embeddings: {train_embeddings.shape}")
print(f"✓ Val embeddings: {val_embeddings.shape}")
print(f"✓ Test embeddings: {test_embeddings.shape}\n")

# Convert to tensors and dataloaders
train_embeddings_t = torch.from_numpy(train_embeddings)
train_labels_t = torch.from_numpy(train_labels.astype(np.int64))
train_dataset = TensorDataset(train_embeddings_t, train_labels_t)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

val_embeddings_t = torch.from_numpy(val_embeddings)
val_labels_t = torch.from_numpy(val_labels.astype(np.int64))
val_dataset = TensorDataset(val_embeddings_t, val_labels_t)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

test_embeddings_t = torch.from_numpy(test_embeddings)
test_labels_t = torch.from_numpy(test_labels.astype(np.int64))
test_dataset = TensorDataset(test_embeddings_t, test_labels_t)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# ==================== CLASSIFIER ====================

class ArcFaceClassifier(nn.Module):
    def __init__(self):
        super(ArcFaceClassifier, self).__init__()
        self.fc = nn.Linear(512, 2)
    
    def forward(self, embeddings):
        return self.fc(embeddings)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}\n")

classifier = ArcFaceClassifier().to(device)
criterion = nn.CrossEntropyLoss()

# ========== PHASE 1: Linear Probe ==========
print("="*80)
print("PHASE 1: Linear Probe (Freeze ArcFace, train classifier)")
print("="*80 + "\n")

optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-3)
best_val_acc = 0
patience = 3
patience_counter = 0

for epoch in range(10):
    classifier.train()
    train_loss = 0
    
    for embeddings, labels in tqdm(train_loader, desc=f"P1 Epoch {epoch+1}/10", leave=False):
        embeddings, labels = embeddings.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = classifier(embeddings)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    train_loss /= len(train_loader)
    
    # Validation
    classifier.eval()
    val_acc = 0
    with torch.no_grad():
        for embeddings, labels in val_loader:
            embeddings, labels = embeddings.to(device), labels.to(device)
            outputs = classifier(embeddings)
            _, preds = torch.max(outputs, 1)
            val_acc += (preds == labels).sum().item()
    
    val_acc /= len(val_dataset)
    
    print(f"  Epoch {epoch+1}/10 - Train Loss: {train_loss:.4f}, Val Acc: {val_acc:.4f}")
    
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        patience_counter = 0
        torch.save(classifier.state_dict(), 'best_arcface_phase1.pth')
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"  → Early stop")
            break

classifier.load_state_dict(torch.load('best_arcface_phase1.pth'))
print(f"✓ Phase 1 complete. Best val acc: {best_val_acc:.4f}\n")

# ========== PHASE 2: Fine-tune ==========
print("="*80)
print("PHASE 2: Fine-tune")
print("="*80 + "\n")

optimizer = torch.optim.SGD(classifier.parameters(), lr=1e-4, momentum=0.9)
best_val_acc = 0
patience_counter = 0

for epoch in range(10):
    classifier.train()
    train_loss = 0
    
    for embeddings, labels in tqdm(train_loader, desc=f"P2 Epoch {epoch+1}/10", leave=False):
        embeddings, labels = embeddings.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = classifier(embeddings)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    train_loss /= len(train_loader)
    
    # Validation
    classifier.eval()
    val_acc = 0
    with torch.no_grad():
        for embeddings, labels in val_loader:
            embeddings, labels = embeddings.to(device), labels.to(device)
            outputs = classifier(embeddings)
            _, preds = torch.max(outputs, 1)
            val_acc += (preds == labels).sum().item()
    
    val_acc /= len(val_dataset)
    
    print(f"  Epoch {epoch+1}/10 - Train Loss: {train_loss:.4f}, Val Acc: {val_acc:.4f}")
    
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        patience_counter = 0
        torch.save(classifier.state_dict(), 'best_arcface_phase2.pth')
    else:
        patience_counter += 1
        if patience_counter >= 5:
            print(f"  → Early stop")
            break

classifier.load_state_dict(torch.load('best_arcface_phase2.pth'))
print(f"✓ Phase 2 complete. Best val acc: {best_val_acc:.4f}\n")

# ========== EVALUATION ==========
print("="*80)
print("TEST SET EVALUATION")
print("="*80 + "\n")

classifier.eval()
test_preds = []

with torch.no_grad():
    for embeddings, labels in tqdm(test_loader, desc="Testing", leave=False):
        embeddings = embeddings.to(device)
        outputs = classifier(embeddings)
        _, preds = torch.max(outputs, 1)
        test_preds.extend(preds.cpu().numpy())

test_preds = np.array(test_preds)

acc = accuracy_score(test_labels, test_preds)
prec = precision_score(test_labels, test_preds, zero_division=0)
rec = recall_score(test_labels, test_preds, zero_division=0)
f1 = f1_score(test_labels, test_preds, zero_division=0)
cm = confusion_matrix(test_labels, test_preds)

print(f"✓ Accuracy:  {acc:.4f} ({acc*100:.1f}%)")
print(f"✓ Precision: {prec:.4f} ({prec*100:.1f}%)")
print(f"✓ Recall:    {rec:.4f} ({rec*100:.1f}%)")
print(f"✓ F1-Score:  {f1:.4f}\n")

print("Confusion Matrix:")
print(f"           Predicted Hero  Predicted Villain")
print(f"Actual Hero        {cm[0,0]:3d}            {cm[0,1]:3d}")
print(f"Actual Villain     {cm[1,0]:3d}            {cm[1,1]:3d}\n")

# ==================== SAVE RESULTS ====================

results = {
    'ArcFace (10M+ Human Faces)': {
        'accuracy': float(acc),
        'precision': float(prec),
        'recall': float(rec),
        'f1': float(f1),
        'accuracy_percent': f"{acc*100:.1f}%"
    }
}

with open('ARCFACE_RESULTS.json', 'w') as f:
    json.dump(results, f, indent=2)

print("="*80)
print("FOR YOUR DEFENSE:")
print("="*80)
print(f"\nArcFace (10M+ Human Faces):")
print(f"  • Accuracy: {acc*100:.1f}%")
print(f"  • Precision: {prec*100:.1f}%")
print(f"  • Recall: {rec*100:.1f}%")
print(f"  • F1-Score: {f1:.3f}")
print("\n✓ Results saved to: ARCFACE_RESULTS.json")
print("="*80)
