# Training 10–15+ Crops (Deep Learning)

## Current setup
- **archive (4)**: 9 crops (Apple, Bell Pepper, Cherry, Corn (Maize), Grape, Peach, Potato, Strawberry, Tomato) — already used by `leaf_model.py`.

## Add 10–15 more crops (train / validate / test)

### Option A: Use “New Plant Diseases Dataset” (14 crops, 38 classes)

1. **Download**  
   [New Plant Diseases Dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset) from Kaggle (train + valid with 38 class folders).

2. **Set paths** in `reorganize_plant_diseases_dataset.py`:
   - `SOURCE_DIR` = path to the unzipped dataset folder (contains `train`, `valid`).
   - `OUT_DIR` = `archive_extra_crops` (default: inside project).

3. **Reorganize**  
   Run:
   ```bash
   python reorganize_plant_diseases_dataset.py
   ```
   This builds per-crop folders: `CropName/Train`, `CropName/Val`, `CropName/Test` (e.g. Tomato, Potato, Corn, Pepper, etc.).

4. **Train / validate / test all crops**  
   Run:
   ```bash
   python leaf_model_multi_crop.py
   ```
   - Uses **archive (4)** (9 crops) and, if present, **archive_extra_crops** (e.g. 14 crops from New Plant Diseases).
   - For each crop: **train** (5 epochs), **validate**, **test**, classification report + confusion matrix, then save `leaf_classifier_<crop>.h5` and `leaf_classes_<crop>.json`.

5. **UI**  
   The app already includes Blueberry, Orange, Pepper, Raspberry, Soybean, Squash. After training, those models are used automatically when you select the crop.

### Option B: Your own dataset (same folder structure)

- Put each crop in a folder with **Train**, **Val**, and **Test** subfolders; inside each, one subfolder per class (e.g. `Healthy`, `Early Blight`).
- Place that folder next to the project or set its path in `leaf_model_multi_crop.py` → `DATA_ROOTS` or `EXTRA_DATA_ROOT`.
- Run `python leaf_model_multi_crop.py`. It will **discover** all crops that have Train/Val/Test and train them.
- Add new crop names to `CROP_MODEL_CONFIG` (and optionally `CROP_SENSITIVITY`) in `app.py` so they appear in the dropdown.

### Summary
| Step              | Script / action                                  |
|-------------------|--------------------------------------------------|
| Get 14 more crops | Download New Plant Diseases Dataset (Kaggle)    |
| Reorganize        | `python reorganize_plant_diseases_dataset.py`   |
| Train/val/test    | `python leaf_model_multi_crop.py`                |
| Use in app        | Crop list in app already includes extra crops   |
