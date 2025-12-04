# Changelog

## File Renaming (December 2024)

All scripts have been renamed to professional, descriptive names for public release.

### Data Collection Scripts
- `hero.py` → `download_hero_images.py`
- `anti.py` → `download_villain_images.py`

### Training Scripts
- `train.py` → `train_resnet_design_features.py`
- `TrainCnn.py` → `train_clip_classifier.py`

### Model Variant Scripts
- `vegg.py` → `vggface2_transfer_learning.py`
- `pls.py` → `vggface2_baseline.py`
- `pls_2.py` → `vggface2_variant1.py`
- `pls_3.py` → `vggface2_variant2.py`
- `pls_4.py` → `vggface2_variant3.py`
- `arcface.py` → `arcface_baseline.py`
- `arcface_1.py` → `arcface_variant1.py`
- `arcface_2.py` → `arcface_variant2.py`
- `advanced.py` → `face_models_comparison.py`
- `advanced_2.py` → `face_models_comparison_v2.py`
- `advanced_face.py` → `vggface_only_classifier.py`

## Project Reorganization (December 2024)

### Directory Structure
- Created organized folder structure:
  - `scripts/` - All Python scripts organized by purpose
  - `data/` - All data files (images and metadata)
  - `models/` - Model weights and checkpoints
  - `results/` - All outputs organized by type
  - `docs/` - Documentation files

### Path Updates
- Updated all script paths to use relative paths from project root
- All scripts now save outputs to organized directories
- Data collection scripts updated to save to `data/images/`

### Documentation
- Created professional, publication-ready README.md
- Added comprehensive project structure documentation
- Added changelog for tracking changes

