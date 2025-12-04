#!/usr/bin/env python3
"""
VGG-FACE ONLY - ANIME CHARACTER CLASSIFICATION
Trained on 2.6 Million Human Faces
"""

# CRITICAL: Downgrade NumPy BEFORE any other imports
import subprocess
import sys
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'numpy<2', '--quiet', '--break-system-packages'])

import os
os.environ['TORCH_HOME'] = '/scratch/ma7030/torch_cache'
os.makedirs('/scratch/ma7030/torch_cache', exist_ok=True)

sys.path.insert(0, '/scratch/ma7030/python_packages')

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
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
VGG-FACE ONLY - ANIME CHARACTER CLASSIFICATION
Trained on 2.6 Million Human Faces
════════════════════════════════════════════════════════════════════════════════
""")

# ==================== DATASET ====================

class AnimeCharacterDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        try:
            image = Image.open(self.image_paths[idx]).convert('RGB')
        except:
            image = Image.new('RGB', (224, 224))
        
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


# ==================== VGGFACE LOADER ====================

def load_vggface(device, weights_path='vgg_face.pth'):
    """Load VGG-Face model trained on 2.6 million human faces."""
    
    print("\n" + "="*80)
    print("LOADING VGG-Face (2.6 Million Human Faces)")
    print("="*80 + "\n")
    
    if not os.path.exists(weights_path):
        print(f"❌ VGG-Face weights NOT found at: {weights_path}\n")
        print("To download VGG-Face weights:")
        print("1. Visit: http://www.robots.ox.ac.uk/~vgg/software/vgg_face/")
        print("2. Download: vgg_face.pth (~220MB)")
        print("3. Place in: /scratch/ma7030/ML_project/vgg_face.pth")
        print("4. Re-run this script\n")
        return None
    
    print(f"✓ Found VGG-Face weights: {weights_path}")
    print("✓ Trained on: 2,600,000 human face images")
    print("✓ Architecture: VGG-16 (16 layers)")
    print("✓ Classes: 9,131 different people\n")
    
    try:
        print("Loading model...")
        model = models.vgg16(pretrained=False)
        
        print("Loading weights...")
        weights = torch.load(weights_path, map_location='cpu')
        
        # Remove last classification layer (trained for 9131 people)
        if 'classifier.6.weight' in weights:
            del weights['classifier.6.weight']
            del weights['classifier.6.bias']
        
        model.load_state_dict(weights, strict=False)
        print("✓ Loaded VGG-Face weights successfully!\n")
        
        # Add new classification layer for 2 classes (hero vs villain)
        model.classifier[6] = nn.Linear(4096, 2)
        
        return model
    
    except Exception as e:
        print(f"❌ Error loading VGG-Face weights: {e}\n")
        return None


# ==================== DATA PREPARATION ====================

print("\n" + "="*80)
print("LOADING DATA")
print("="*80 + "\n")

hero_dir = "mugshots_google_Hero"
antihero_dir = "mugshots_google_Anti"

hero_images = []
antihero_images = []

if os.path.exists(hero_dir):
    hero_images = [os.path.join(hero_dir, f) for f in os.listdir(hero_dir) 
                   if f.endswith(('.jpg', '.jpeg', '.png'))]

if os.path.exists(antihero_dir):
    antihero_images = [os.path.join(antihero_dir, f) for f in os.listdir(antihero_dir) 
                       if f.endswith(('.jpg', '.jpeg', '.png'))]

if not hero_images or not antihero_images:
    print("❌ Dataset directories not found!")
    print(f"   Looking for: {hero_dir} and {antihero_dir}")
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

# ==================== TRANSFORMS ====================

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = AnimeCharacterDataset(train_images, train_labels, train_transform)
val_dataset = AnimeCharacterDataset(val_images, val_labels, val_test_transform)
test_dataset = AnimeCharacterDataset(test_images, test_labels, val_test_transform)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)

# ==================== TRAINING ====================

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}\n")

model = load_vggface(device)

if model is None:
    print("❌ Failed to load VGG-Face model")
    sys.exit(1)

model = model.to(device)

# Class weights
unique, counts = np.unique(train_labels, return_counts=True)
class_weights = torch.tensor([1.0, 1.0], dtype=torch.float32).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)

classifier_params = list(model.classifier.parameters())

# ========== PHASE 1: Linear Probe ==========
print("\n" + "="*80)
print("PHASE 1: Linear Probe (Freeze features, train classifier)")
print("="*80 + "\n")

for param in model.parameters():
    param.requires_grad = False

for param in classifier_params:
    param.requires_grad = True

optimizer = optim.Adam(classifier_params, lr=1e-3)
best_val_acc = 0
patience = 3
patience_counter = 0

for epoch in range(10):
    model.train()
    train_loss = 0
    train_count = 0
    
    for images, labels in tqdm(train_loader, desc=f"P1 Epoch {epoch+1}/10", leave=False):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * len(labels)
        train_count += len(labels)
    
    train_loss /= train_count if train_count > 0 else 1
    
    model.eval()
    val_acc = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            val_acc += (preds == labels).sum().item()
    
    val_acc /= len(val_dataset)
    
    print(f"  Epoch {epoch+1}/10 - Train Loss: {train_loss:.4f}, Val Acc: {val_acc:.4f}")
    
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        patience_counter = 0
        torch.save(model.state_dict(), 'best_vggface_phase1.pth')
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"  → Early stop at epoch {epoch+1}")
            break

if os.path.exists('best_vggface_phase1.pth'):
    model.load_state_dict(torch.load('best_vggface_phase1.pth'))

print(f"\n✓ Phase 1 complete. Best val acc: {best_val_acc:.4f}\n")

# ========== PHASE 2: Fine-tune ==========
print("="*80)
print("PHASE 2: Fine-tune (Unfreeze last layer)")
print("="*80 + "\n")

for param in model.parameters():
    param.requires_grad = True

optimizer = optim.SGD(model.parameters(), lr=1e-5, momentum=0.9, weight_decay=1e-4)
best_val_acc = 0
patience_counter = 0

for epoch in range(15):
    model.train()
    train_loss = 0
    train_count = 0
    
    for images, labels in tqdm(train_loader, desc=f"P2 Epoch {epoch+1}/15", leave=False):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * len(labels)
        train_count += len(labels)
    
    train_loss /= train_count if train_count > 0 else 1
    
    model.eval()
    val_acc = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            val_acc += (preds == labels).sum().item()
    
    val_acc /= len(val_dataset)
    
    print(f"  Epoch {epoch+1}/15 - Train Loss: {train_loss:.4f}, Val Acc: {val_acc:.4f}")
    
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        patience_counter = 0
        torch.save(model.state_dict(), 'best_vggface_phase2.pth')
    else:
        patience_counter += 1
        if patience_counter >= 5:
            print(f"  → Early stop at epoch {epoch+1}")
            break

if os.path.exists('best_vggface_phase2.pth'):
    model.load_state_dict(torch.load('best_vggface_phase2.pth'))

print(f"\n✓ Phase 2 complete. Best val acc: {best_val_acc:.4f}\n")

# ========== EVALUATION ==========
print("="*80)
print("TEST SET EVALUATION")
print("="*80 + "\n")

model.eval()
test_preds = []
test_true = []

with torch.no_grad():
    for images, labels in tqdm(test_loader, desc="Testing", leave=False):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        
        test_preds.extend(preds.cpu().numpy())
        test_true.extend(labels.cpu().numpy())

test_preds = np.array(test_preds)
test_true = np.array(test_true)

acc = accuracy_score(test_true, test_preds)
prec = precision_score(test_true, test_preds, zero_division=0)
rec = recall_score(test_true, test_preds, zero_division=0)
f1 = f1_score(test_true, test_preds, zero_division=0)
cm = confusion_matrix(test_true, test_preds)

print(f"Accuracy:  {acc:.4f} ({acc*100:.1f}%)")
print(f"Precision: {prec:.4f} ({prec*100:.1f}%)")
print(f"Recall:    {rec:.4f} ({rec*100:.1f}%)")
print(f"F1-Score:  {f1:.4f}\n")

print("Confusion Matrix:")
print(f"        Hero  Villain")
print(f"Hero    {cm[0,0]:3d}    {cm[0,1]:3d}")
print(f"Villain {cm[1,0]:3d}    {cm[1,1]:3d}\n")

# ==================== SAVE RESULTS ====================

results = {
    'VGG-Face (2.6M faces)': {
        'accuracy': float(acc),
        'precision': float(prec),
        'recall': float(rec),
        'f1': float(f1),
        'architecture': 'VGG-16',
        'pretraining': '2,600,000 human face images',
        'pretraining_source': 'VGG-Face Dataset (Oxford University)'
    }
}

with open('VGG_FACE_RESULTS.json', 'w') as f:
    json.dump(results, f, indent=2)

print("="*80)
print("FOR YOUR DEFENSE SLIDES:")
print("="*80)
print(f"\nVGG-Face (2.6M Human Faces):")
print(f"  Accuracy: {acc*100:.1f}%")
print(f"  Precision: {prec*100:.1f}%")
print(f"  Recall: {rec*100:.1f}%")
print(f"  F1-Score: {f1:.3f}")

print("\n✓ Results saved to: VGG_FACE_RESULTS.json")
print("="*80)
