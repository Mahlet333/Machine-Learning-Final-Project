# Computational Analysis of Visual Design Conventions in Anime Character Classification

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Latest-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-Research-green.svg)](LICENSE)

A comprehensive machine learning framework for classifying anime characters as heroes or villains/antiheroes using multiple deep learning approaches. This project investigates visual design conventions in anime character design through computational analysis, exploring the effectiveness of handcrafted design features versus learned representations from pretrained vision models.

## Abstract

This repository presents a systematic investigation into the visual design conventions that distinguish protagonist (hero) and antagonist (villain/antihero) characters in anime. We implement and compare multiple classification approaches:

1. **Design Convention-Based Features**: 50+ handcrafted features derived from anime design principles (color palettes, texture, edge characteristics, shadow distribution)
2. **CLIP-Based Classification**: Semantic feature extraction using CLIP ViT-L/14
3. **Face-Trained Transfer Learning**: VGGFace2, ArcFace, and other models pretrained on human face datasets
4. **Hybrid Approaches**: Combining design features with deep learning architectures

Our findings indicate that villain design exhibits greater visual distinctiveness than hero design, with color palettes (warm vs. cool), edge sharpness, and shadow density serving as key discriminative features.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Methodology](#methodology)
- [Results](#results)
- [Reproducibility](#reproducibility)
- [Citation](#citation)
- [License](#license)

## Features

- **Multiple Classification Approaches**: Design-based features, CLIP embeddings, and face-trained transfer learning
- **Comprehensive Feature Engineering**: 50+ handcrafted features capturing color, texture, geometry, and higher-level design patterns
- **Robust Evaluation**: Extensive metrics including accuracy, precision, recall, F1-score, AUC-ROC, and feature importance analysis
- **Reproducible Research**: Fixed random seeds, documented data splits, and complete codebase
- **Well-Organized Codebase**: Clean project structure with separated scripts, data, models, and results

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended) or CPU
- 8GB+ RAM recommended
- 5GB+ disk space for models and data

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd Machine_learning
```

### Step 2: Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install torch torchvision torchaudio
pip install numpy pandas scikit-learn
pip install matplotlib seaborn
pip install pillow opencv-python
pip install tqdm open-clip-torch shap scipy
pip install serpapi python-dotenv  # python-dotenv for environment variables
```

**Note**: Some scripts require NumPy < 2.0. If you encounter compatibility issues:
```bash
pip install "numpy<2"
```

### Step 4: Configure Environment Variables

For data collection scripts, you need to set up your SerpAPI key:

1. **Get a SerpAPI Key**: Sign up at [serpapi.com](https://serpapi.com) and get your free API key

2. **Create `.env` file**: Copy the example file and add your API key:
   ```bash
   cp .env.example .env
   ```

3. **Edit `.env` file**: Replace `your_serpapi_key_here` with your actual SerpAPI key:
   ```bash
   # .env
   SERPAPI_KEY=your_actual_api_key_here
   ```

**Note**: The `.env` file is already in `.gitignore` and will not be committed to the repository.

### Step 5: Verify Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "from dotenv import load_dotenv; print('python-dotenv installed successfully')"
```

## Quick Start

### Basic Usage

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run design-based classification (ResNet18)
python scripts/training/train_resnet_design_features.py

# 3. Run CLIP-based classification
python scripts/training/train_clip_classifier.py
```

### Expected Outputs

- **Model Weights**: `models/checkpoints/best_resnet18.pth`, `models/checkpoints/best_clip_classifier.pth`
- **Metrics**: `results/metrics/test_metrics.json`
- **Visualizations**: `results/visualizations/*.png`
- **Feature Analysis**: `results/features/*.csv`

## Project Structure

```
Machine_learning/
├── scripts/                          # Python scripts
│   ├── data_collection/            # Data acquisition scripts
│   │   ├── download_hero_images.py
│   │   └── download_villain_images.py
│   ├── training/                    # Main training pipelines
│   │   ├── train_resnet_design_features.py  # Design features + ResNet18
│   │   └── train_clip_classifier.py         # CLIP ViT-L/14 + MLP
│   └── models/                       # Model variant experiments
│       ├── vggface2_transfer_learning.py
│       ├── vggface2_baseline.py
│       ├── vggface2_variant*.py
│       ├── arcface_baseline.py
│       ├── arcface_variant*.py
│       ├── face_models_comparison.py
│       └── vggface_only_classifier.py
│
├── data/                            # Data files
│   ├── images/                      # Image datasets
│   │   ├── hero/                    # Hero character images (150)
│   │   └── villain/                 # Villain/antihero images (150)
│   └── metadata/                    # Character metadata
│       ├── Hero_list.csv
│       └── Anti_Hero.csv
│
├── models/                          # Model weights
│   ├── checkpoints/                 # Trained model checkpoints
│   └── pretrained/                  # Pretrained model files
│
├── results/                         # Output files
│   ├── features/                   # Feature files and CSVs
│   ├── visualizations/             # Visualization PNGs
│   └── metrics/                     # Evaluation metrics and reports
│
├── docs/                            # Documentation
│   └── STRUCTURE.md                 # Detailed structure guide
│
├── README.md                        # This file
└── requirements.txt                 # Python dependencies
```

## Usage

### Data Collection

To collect new character images:

1. **Set up your API key** (if not already done):
   - Create a `.env` file in the project root (copy from `.env.example`)
   - Add your SerpAPI key: `SERPAPI_KEY=your_actual_api_key_here`
   - Get your API key from [serpapi.com](https://serpapi.com)

2. **Run collection scripts**:
   ```bash
   python scripts/data_collection/download_hero_images.py
   python scripts/data_collection/download_villain_images.py
   ```

Images will be downloaded, cropped to 256×256 square mugshots, and saved to `data/images/hero/` and `data/images/villain/`.

**Note**: The scripts will automatically load your API key from the `.env` file. No need to modify the script files.

### Training Models

#### Design-Based Classification

Trains ResNet18 using 50+ handcrafted design features:

```bash
python scripts/training/train_resnet_design_features.py
```

**Process**:
1. Extracts design features (color, texture, edges, shadows)
2. Labels images based on design conventions
3. Trains ResNet18 in two phases: linear probe → fine-tuning
4. Evaluates on test set and generates feature importance analysis

**Outputs**:
- Model weights: `models/checkpoints/best_resnet18.pth`
- Metrics: `results/metrics/test_metrics.json`
- Feature rankings: `results/features/design_feature_importance.csv`
- Research report: `results/metrics/ANIME_DESIGN_RESEARCH_REPORT.txt`

#### CLIP-Based Classification

Uses CLIP ViT-L/14 for semantic feature extraction:

```bash
python scripts/training/train_clip_classifier.py
```

**Process**:
1. Extracts CLIP features from all images
2. Trains 3-layer MLP classifier
3. Performs feature importance analysis (Logistic Regression, Random Forest, SHAP)
4. Generates comprehensive visualizations

**Outputs**:
- Model weights: `models/checkpoints/best_clip_classifier.pth`
- Features: `results/features/clip_features.npy`, `results/features/clip_labels.npy`
- Feature importance: `results/features/feature_importance_table.csv`
- Visualizations: Training curves, confusion matrix, ROC curve, feature importance plots

#### Face-Trained Transfer Learning

Experiments with VGGFace2 and ArcFace models:

```bash
# VGGFace2 transfer learning
python scripts/models/vggface2_transfer_learning.py

# ArcFace baseline
python scripts/models/arcface_baseline.py

# Face models comparison
python scripts/models/face_models_comparison.py
```

## Methodology

### Design Feature Extraction

The design-based approach extracts 50+ features across four categories:

1. **Pixel-Level & Color Features** (19 features):
   - RGB statistics (mean, std)
   - HSV color space metrics
   - Brightness and contrast measures
   - Color warmth indicators
   - Saturation levels

2. **Texture and Edge Features** (15 features):
   - Sobel edge detection metrics
   - Edge density and strength
   - Texture variance and entropy
   - Gradient measurements
   - Laplacian features

3. **Shape and Geometric Features** (11 features):
   - Center region brightness
   - Spatial brightness distribution
   - Roundness and angularity scores

4. **Higher-Level Features** (5+ features):
   - Shadow density and concentration
   - Aura brightness/darkness
   - Scar/mark likelihood

### Model Architectures

- **ResNet18**: Pretrained on ImageNet, fine-tuned for binary classification
- **CLIP ViT-L/14**: Pretrained vision-language model, features extracted and classified via MLP
- **VGGFace2**: VGG16 backbone pretrained on 3.31M human faces
- **ArcFace**: Face recognition model from InsightFace

### Training Procedure

- **Data Split**: 60% train, 20% validation, 20% test (stratified)
- **Augmentation**: Random horizontal flip, rotation (±15°), color jitter
- **Loss Function**: Weighted CrossEntropyLoss to handle class imbalance
- **Optimization**: Adam (linear probe), SGD with momentum (fine-tuning)
- **Early Stopping**: Based on validation accuracy with patience

## Results

### Performance Summary

| Method | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
|--------|----------|-----------|--------|----------|---------|
| Design Features + ResNet18 | ~80% | ~80% | ~100% | ~89% | ~0.90 |
| CLIP ViT-L/14 + MLP | Variable | Variable | Variable | Variable | Variable |
| VGGFace2 Transfer Learning | ~66.7% | ~75% | ~83% | ~60% | Variable |

### Key Findings

1. **Villain Design Distinctiveness**: Villain characters exhibit more visually distinctive design patterns than heroes, leading to higher recall for villain classification.

2. **Color Palette Significance**: Warm (red/orange) vs. cool (blue/purple) color palettes serve as strong discriminative features.

3. **Edge and Shadow Characteristics**: Sharp edges and high shadow density correlate strongly with villain classification.

4. **Domain Transfer Effectiveness**: Face-trained models (VGGFace2, ArcFace) show promise for anime character classification, suggesting domain similarity between human faces and stylized anime faces.

5. **Feature Engineering Value**: Handcrafted design features achieve competitive performance compared to learned representations, indicating the value of domain knowledge in feature engineering.

## Reproducibility

### Complete Reproduction Steps

1. **Environment Setup**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Data Verification**:
   - Ensure `data/images/hero/` and `data/images/villain/` contain ~150 images each
   - Verify CSV files in `data/metadata/` are present

3. **Run Training**:
   ```bash
   python scripts/training/train_resnet_design_features.py
   python scripts/training/train_clip_classifier.py
   ```

4. **Verify Outputs**:
   - Check `models/checkpoints/` for saved model weights
   - Review `results/metrics/` for evaluation metrics
   - Examine `results/visualizations/` for plots

### Reproducibility Guarantees

- **Random Seeds**: All scripts use `random_state=42` for deterministic results
- **Data Splits**: Fixed 60/20/20 train/val/test split with stratification
- **Model Initialization**: Pretrained weights downloaded automatically
- **Hardware Compatibility**: Scripts work on both CPU and GPU (CUDA)

### Troubleshooting

**OpenCV Threading Issues**:
- Scripts automatically disable OpenCV threading (`cv2.setNumThreads(1)`)
- If issues persist, verify environment variables are set correctly

**Memory Errors**:
- Reduce batch size in DataLoader (default: 16 → 8 or 4)
- Use CPU instead of GPU if VRAM is limited

**NumPy Compatibility**:
- Install NumPy < 2.0: `pip install "numpy<2"`

**Missing Pretrained Models**:
- Models download automatically on first run
- Ensure stable internet connection

**CUDA Out of Memory**:
- Reduce batch size or image resolution
- Use gradient accumulation for effective larger batch sizes

This project leveraged the NYU High Performance Computing (HPC) cluster for large-scale model training, feature extraction, and experimentation. The HPC environment enabled:

- Efficient batch processing for CLIP feature extraction

- Faster training and fine-tuning of deep CNN models

- Large-scale hyperparameter experiments for transfer learning models (VGGFace2, ArcFace)

- Parallel execution of feature engineering pipelines and evaluation scripts

Using NYU’s HPC infrastructure significantly reduced training time and made large, multi-model comparisons computationally feasible.

## Citation

If you use this code or findings in your research, please cite:

```bibtex
@software{anime_character_classification,
  title = {Computational Analysis of Visual Design Conventions in Anime Character Classification},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/yourusername/Machine_learning}
}
```

## License

This project is released for research and educational purposes. Please refer to the LICENSE file for details.

## Acknowledgments

- CLIP model: [OpenAI CLIP](https://github.com/openai/CLIP)
- VGGFace2: [VGGFace2 Dataset](https://www.robots.ox.ac.uk/~vgg/data/vgg_face2/)
- ArcFace: [InsightFace](https://github.com/deepinsight/insightface)
- PyTorch: [PyTorch](https://pytorch.org/)

## Contact

For questions, issues, or collaboration inquiries, please open an issue on GitHub or contact the repository maintainer.

---
**Group Members name**:
1. Mahlet Atrsaw Andargei
2. Eyerusalem Hawoltu Afework
**Last Updated**: December 2025  
**Python Version**: 3.8+  
**PyTorch Version**: Latest stable  
**Dataset Size**: ~300 images (150 heroes, 150 villains/antiheroes)
