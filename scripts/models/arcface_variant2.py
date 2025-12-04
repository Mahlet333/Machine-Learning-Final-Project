#!/usr/bin/env python3
"""
ARCFACE TRAINING - MAXIMUM PERFORMANCE
- Forces GPU execution
- Adds AUC-ROC metric
- Data augmentation via mixup
- Better handling of imbalanced data
"""

import subprocess
import sys
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'numpy<2', '--quiet', '--break-system-packages'])

import os
os.environ['INSIGHTFACE_DATA_HOME'] = '/scratch/ma7030/insightface_cache'
os.environ['TORCH_HOME'] = '/scratch/ma7030/torch_cache'
sys.path.insert(0, '/scratch/ma7030/python_packages')

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix
from sklearn.preprocessing import StandardScaler
from PIL import Image
import json
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

print("""
════════════════════════════════════════════════════════════════════════════════
ARCFACE TRAINING - MAXIMUM PERFORMANCE
GPU Forced + AUC + Data Augmentation + Imbalance Handling
════════════════════════════════════════════════════════════════════════════════
""")

# ==================== LOAD ARCFACE - FORCE GPU ====================

print("\n" + "="*80)
print("LOADING ARCFACE (FORCE GPU MODE)")
print("="*80)

try:
    # Import and configure CUDA
    import torch
    print(f"✓ CUDA available: {torch.cuda.is_available()}")
    print(f"✓ CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}\n")
    
    from insightface.app import FaceAnalysis
    
    # FORCE CUDA - try ONNX Runtime with CUDA first
    arcface_app = FaceAnalysis(
        name='buffalo_l',
        root='/scratch/ma7030/insightface_cache',
        providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
    )
    
    # Prepare with GPU (ctx_id=0)
    arcface_app.prepare(ctx_id=0)
    
    print("✓ ArcFace forced to GPU (ctx_id=0)")
    print("✓ Extracting embeddings will use GPU\n")
    
except Exception as e:
    print(f"Warning: {e}")
    print("Continuing with CPU fallback...\n")
    arcface_app = FaceAnalysis(
        name='buffalo_l',
        root='/scratch/ma7030/insightface_cache',
        providers=['CPUExecutionProvider']
    )
    arcface_app.prepare(ctx_id=-1)

# ==================== DATA PREP ====================

print("="*80)
print("LOADING DATASET")
print("="*80 + "\n")

hero_images = sorted([f"mugshots_google_Hero/{f}" for f in os.listdir("mugshots_google_Hero") 
               if f.endswith(('.jpg', '.jpeg', '.png'))])
antihero_images = sorted([f"mugshots_google_Anti/{f}" for f in os.listdir("mugshots_google_Anti") 
                   if f.endswith(('.jpg', '.jpeg', '.png'))])

all_images = hero_images + antihero_images
all_labels = np.array([0]*len(hero_images) + [1]*len(antihero_images))

print(f"Heroes: {len(hero_images)}")
print(f"Villains: {len(antihero_images)}")
print(f"Total: {len(all_images)}\n")

train_images, temp_images, train_labels, temp_labels = train_test_split(
    all_images, all_labels, test_size=0.4, random_state=42, stratify=all_labels
)
val_images, test_images, val_labels, test_labels = train_test_split(
    temp_images, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels
)

print(f"Train: {len(train_images)} (heroes: {sum(train_labels==0)}, villains: {sum(train_labels==1)})")
print(f"Val: {len(val_images)} (heroes: {sum(val_labels==0)}, villains: {sum(val_labels==1)})")
print(f"Test: {len(test_images)} (heroes: {sum(test_labels==0)}, villains: {sum(test_labels==1)})\n")

# ==================== EXTRACT EMBEDDINGS ====================

print("="*80)
print("EXTRACTING ARCFACE EMBEDDINGS (GPU)")
print("="*80 + "\n")

def extract_embeddings_robust(image_files, app):
    """Extract embeddings with quality checks"""
    embeddings = []
    success_count = 0
    
    for img_path in tqdm(image_files, desc="Extracting", leave=False):
        try:
            img = Image.open(img_path).convert('RGB')
            img_np = np.array(img)
            
            # Validate image
            if img_np.shape[0] < 50 or img_np.shape[1] < 50:
                embeddings.append(np.random.randn(512) * 0.01)
                continue
            
            faces = app.get(img_np)
            if len(faces) > 0 and faces[0].embedding is not None:
                emb = faces[0].embedding
                # Validate embedding
                if not np.isnan(emb).any() and not np.isinf(emb).any():
                    embeddings.append(emb)
                    success_count += 1
                else:
                    embeddings.append(np.random.randn(512) * 0.01)
            else:
                embeddings.append(np.random.randn(512) * 0.01)
        except:
            embeddings.append(np.random.randn(512) * 0.01)
    
    print(f"Successfully extracted: {success_count}/{len(image_files)}")
    return np.array(embeddings, dtype=np.float32)

train_emb = extract_embeddings_robust(train_images, arcface_app)
val_emb = extract_embeddings_robust(val_images, arcface_app)
test_emb = extract_embeddings_robust(test_images, arcface_app)

print(f"✓ Shapes - Train: {train_emb.shape}, Val: {val_emb.shape}, Test: {test_emb.shape}\n")

# ==================== NORMALIZE ====================

print("Normalizing embeddings with StandardScaler...")
scaler = StandardScaler()
train_emb_norm = scaler.fit_transform(train_emb)
val_emb_norm = scaler.transform(val_emb)
test_emb_norm = scaler.transform(test_emb)

train_emb = torch.from_numpy(train_emb_norm).float()
val_emb = torch.from_numpy(val_emb_norm).float()
test_emb = torch.from_numpy(test_emb_norm).float()

train_labels_t = torch.from_numpy(train_labels.astype(np.int64))
val_labels_t = torch.from_numpy(val_labels.astype(np.int64))
test_labels_t = torch.from_numpy(test_labels.astype(np.int64))

print("✓ Normalized\n")

# ==================== DATA AUGMENTATION - MIXUP ====================

def mixup_batch(x, y, alpha=0.2):
    """Mixup data augmentation"""
    batch_size = x.size(0)
    index = torch.randperm(batch_size)
    lam = np.random.beta(alpha, alpha)
    
    mixed_x = lam * x + (1 - lam) * x[index, :]
    mixed_y = lam * y.float() + (1 - lam) * y[index].float()
    
    return mixed_x, mixed_y

# ==================== DATALOADERS ====================

train_loader = DataLoader(
    TensorDataset(train_emb, train_labels_t),
    batch_size=16, shuffle=True
)
val_loader = DataLoader(
    TensorDataset(val_emb, val_labels_t),
    batch_size=32, shuffle=False
)
test_loader = DataLoader(
    TensorDataset(test_emb, test_labels_t),
    batch_size=32, shuffle=False
)

# ==================== MODEL ====================

class HighPerformanceClassifier(nn.Module):
    """Optimized classifier with residual-style connections"""
    def __init__(self, input_dim=512):
        super().__init__()
        
        self.fc1 = nn.Linear(input_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.act1 = nn.GELU()
        self.drop1 = nn.Dropout(0.4)
        
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.act2 = nn.GELU()
        self.drop2 = nn.Dropout(0.3)
        
        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.act3 = nn.GELU()
        self.drop3 = nn.Dropout(0.2)
        
        self.fc_out = nn.Linear(64, 2)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.drop1(x)
        
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.act2(x)
        x = self.drop2(x)
        
        x = self.fc3(x)
        x = self.bn3(x)
        x = self.act3(x)
        x = self.drop3(x)
        
        x = self.fc_out(x)
        return x

device = torch.device('cuda')
model = HighPerformanceClassifier().to(device)

# Class balancing
class_counts = np.bincount(train_labels)
class_weights = torch.tensor([1.0, class_counts[0]/class_counts[1]], dtype=torch.float32).to(device)
print(f"Class weights: {class_weights.cpu().numpy()}")

criterion = nn.CrossEntropyLoss(weight=class_weights)

# ========== PHASE 1 ==========
print("\n" + "="*80)
print("PHASE 1: Linear Probe")
print("="*80 + "\n")

optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.7)

best_auc = 0
patience = 5
patience_counter = 0

for epoch in range(30):
    model.train()
    train_loss = 0
    
    for emb, lbl in train_loader:
        emb, lbl = emb.to(device), lbl.to(device)
        
        # Apply mixup
        if np.random.rand() > 0.3:
            emb, lbl = mixup_batch(emb, lbl)
        
        optimizer.zero_grad()
        outputs = model(emb)
        loss = criterion(outputs, lbl.long() if lbl.dtype == torch.float32 else lbl)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
    
    # Validation
    model.eval()
    with torch.no_grad():
        val_preds = []
        val_probs = []
        for emb, _ in val_loader:
            emb = emb.to(device)
            outputs = model(emb)
            val_preds.extend(outputs.argmax(1).cpu().numpy())
            val_probs.extend(torch.softmax(outputs, 1)[:, 1].cpu().numpy())
    
    val_acc = accuracy_score(val_labels, val_preds)
    val_auc = roc_auc_score(val_labels, val_probs)
    
    print(f"  E{epoch+1:2d} - Loss: {train_loss/len(train_loader):.4f}, Acc: {val_acc:.4f}, AUC: {val_auc:.4f}")
    
    scheduler.step()
    
    if val_auc > best_auc:
        best_auc = val_auc
        patience_counter = 0
        torch.save(model.state_dict(), 'best_p1.pth')
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"  → Early stop")
            break

model.load_state_dict(torch.load('best_p1.pth'))
print(f"✓ Best AUC: {best_auc:.4f}\n")

# ========== PHASE 2 ==========
print("="*80)
print("PHASE 2: Fine-tune")
print("="*80 + "\n")

optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)

best_auc = 0
patience_counter = 0

for epoch in range(30):
    model.train()
    train_loss = 0
    
    for emb, lbl in train_loader:
        emb, lbl = emb.to(device), lbl.to(device)
        
        if np.random.rand() > 0.5:
            emb, lbl = mixup_batch(emb, lbl, alpha=0.3)
        
        optimizer.zero_grad()
        outputs = model(emb)
        loss = criterion(outputs, lbl.long() if lbl.dtype == torch.float32 else lbl)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
    
    # Validation
    model.eval()
    with torch.no_grad():
        val_preds = []
        val_probs = []
        for emb, _ in val_loader:
            emb = emb.to(device)
            outputs = model(emb)
            val_preds.extend(outputs.argmax(1).cpu().numpy())
            val_probs.extend(torch.softmax(outputs, 1)[:, 1].cpu().numpy())
    
    val_acc = accuracy_score(val_labels, val_preds)
    val_auc = roc_auc_score(val_labels, val_probs)
    
    print(f"  E{epoch+1:2d} - Loss: {train_loss/len(train_loader):.4f}, Acc: {val_acc:.4f}, AUC: {val_auc:.4f}")
    
    scheduler.step()
    
    if val_auc > best_auc:
        best_auc = val_auc
        patience_counter = 0
        torch.save(model.state_dict(), 'best_p2.pth')
    else:
        patience_counter += 1
        if patience_counter >= 8:
            print(f"  → Early stop")
            break

model.load_state_dict(torch.load('best_p2.pth'))
print(f"✓ Best AUC: {best_auc:.4f}\n")

# ========== EVALUATION ==========
print("="*80)
print("TEST SET EVALUATION")
print("="*80 + "\n")

model.eval()
test_preds = []
test_probs = []

with torch.no_grad():
    for emb, _ in test_loader:
        emb = emb.to(device)
        outputs = model(emb)
        test_preds.extend(outputs.argmax(1).cpu().numpy())
        test_probs.extend(torch.softmax(outputs, 1)[:, 1].cpu().numpy())

test_preds = np.array(test_preds)
test_probs = np.array(test_probs)

acc = accuracy_score(test_labels, test_preds)
prec = precision_score(test_labels, test_preds, zero_division=0)
rec = recall_score(test_labels, test_preds, zero_division=0)
f1 = f1_score(test_labels, test_preds, zero_division=0)
auc = roc_auc_score(test_labels, test_probs)
cm = confusion_matrix(test_labels, test_preds)

print(f"✓ Accuracy:  {acc*100:.1f}%")
print(f"✓ Precision: {prec*100:.1f}%")
print(f"✓ Recall:    {rec*100:.1f}%")
print(f"✓ F1-Score:  {f1:.3f}")
print(f"✓ AUC-ROC:   {auc:.3f}\n")

print("Confusion Matrix:")
print(f"           Predicted Hero  Predicted Villain")
print(f"Actual Hero        {cm[0,0]:3d}            {cm[0,1]:3d}")
print(f"Actual Villain     {cm[1,0]:3d}            {cm[1,1]:3d}\n")

# Save
results = {
    'ArcFace (10M+ Human Faces)': {
        'accuracy': float(acc),
        'accuracy_percent': f"{acc*100:.1f}%",
        'precision': float(prec),
        'precision_percent': f"{prec*100:.1f}%",
        'recall': float(rec),
        'recall_percent': f"{rec*100:.1f}%",
        'f1_score': float(f1),
        'auc_roc': float(auc),
        'auc_roc_percent': f"{auc*100:.1f}%"
    }
}

with open('ARCFACE_RESULTS.json', 'w') as f:
    json.dump(results, f, indent=2)

print("="*80)
print("FOR YOUR DEFENSE:")
print("="*80)
print(f"\n🎯 ArcFace (10M+ Human Faces):")
print(f"   Accuracy:  {acc*100:.1f}%")
print(f"   Precision: {prec*100:.1f}%")
print(f"   Recall:    {rec*100:.1f}%")
print(f"   F1-Score:  {f1:.3f}")
print(f"   AUC-ROC:   {auc:.3f}")
print("\n✓ Results saved!")
print("="*80)
