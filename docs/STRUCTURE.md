# Project Structure Guide

This document explains the reorganized folder structure of the Anime Character Classification project.

## Directory Organization

### `/scripts/` - All Python Scripts

All executable Python scripts are organized by purpose:

- **`scripts/data_collection/`** - Scripts for downloading and collecting image data
  - `download_hero_images.py` - Downloads hero character images
  - `download_villain_images.py` - Downloads villain/antihero character images

- **`scripts/training/`** - Main training scripts
  - `train_resnet_design_features.py` - Design features + ResNet18 training
  - `train_clip_classifier.py` - CLIP ViT-L/14 + MLP classifier

- **`scripts/models/`** - Model variant experiments
  - `vggface2_transfer_learning.py` - VGGFace2 transfer learning implementation
  - `vggface2_baseline.py`, `vggface2_variant1.py`, etc. - VGGFace2 variants
  - `arcface_baseline.py`, `arcface_variant1.py`, `arcface_variant2.py` - ArcFace variants
  - `face_models_comparison.py`, `face_models_comparison_v2.py` - Face model comparisons
  - `vggface_only_classifier.py` - VGGFace-only classifier

### `/data/` - All Data Files

- **`data/images/`** - Image datasets
  - `hero/` - Hero character images (150 .jpg files)
  - `villain/` - Villain/antihero character images (150 .jpg files)

- **`data/metadata/`** - Metadata and configuration files
  - `Hero_list.csv` - List of hero characters with anime names
  - `Anti_Hero.csv` - List of villain/antihero characters with anime names

### `/models/` - Model Weights and Checkpoints

- **`models/checkpoints/`** - Trained model weights
  - `best_resnet18.pth` - ResNet18 model weights (from train.py)
  - `best_clip_classifier.pth` - CLIP classifier weights (from TrainCnn.py)
  - `best_phase1.pth` - Phase 1 checkpoint (if exists)

- **`models/pretrained/`** - Pretrained model files
  - `face_paint_512_v2.pt` - Face paint model

### `/results/` - All Output Files

- **`results/features/`** - Feature files and CSVs
  - `clip_features.npy` - Extracted CLIP features
  - `clip_labels.npy` - Labels for CLIP features
  - `feature_importance_table.csv` - Feature importance rankings
  - `design_feature_importance.csv` - Design feature rankings
  - `logistic_importance.csv` - Logistic regression weights
  - `random_forest_importance.csv` - Random forest importances
  - `shap_values.csv` - SHAP values

- **`results/visualizations/`** - All visualization PNGs
  - `training_loss_curve.png` - Training loss over epochs
  - `validation_accuracy_curve.png` - Validation accuracy over epochs
  - `confusion_matrix.png` - Confusion matrix heatmap
  - `roc_curve.png` - ROC curve
  - `logistic_weights_top20.png` - Top 20 logistic regression features
  - `rf_importances_top20.png` - Top 20 random forest features
  - `1_COMPREHENSIVE_ANALYSIS_9PANELS.png` - Comprehensive analysis
  - `2_TRAINING_CURVES.png` - Training curves
  - `3_TRAINING_CURVES_DETAILED.png` - Detailed training curves
  - `4_TRAINING_SUMMARY_TABLE.png` - Training summary table

- **`results/metrics/`** - Evaluation metrics and reports
  - `test_metrics.json` - Test set evaluation metrics
  - `ANIME_DESIGN_RESEARCH_REPORT.txt` - Main research report
  - `SUMMARY_REPORT.json` - Summary report
  - `ARCFACE_RESULTS.json` - ArcFace results
  - `VGGFACE2_*.json` - VGGFace2 variant results
  - `feature_importance_summary.txt` - Feature importance summary
  - `DEFENSE_TALKING_POINTS.txt` - Defense talking points

### `/docs/` - Documentation

- `STRUCTURE.md` - This file (project structure guide)
- Additional documentation can be added here

## Path Updates in Scripts

All scripts have been updated to use relative paths from the project root:

- Data collection scripts now save to `data/images/hero/` and `data/images/villain/`
- Training scripts load from `data/images/` and save to `models/checkpoints/` and `results/`
- All output files are organized into appropriate subdirectories

## Running Scripts

When running scripts, use paths relative to the project root:

```bash
# From project root
python scripts/training/train_resnet_design_features.py
python scripts/training/train_clip_classifier.py
python scripts/data_collection/download_hero_images.py
```

## Benefits of This Structure

1. **Clear Separation**: Scripts, data, models, and results are clearly separated
2. **Easy Navigation**: Related files are grouped together
3. **Scalability**: Easy to add new scripts or results without clutter
4. **Reproducibility**: Clear structure makes it easier to reproduce experiments
5. **Version Control**: Can easily ignore large files (images, models) while tracking code

