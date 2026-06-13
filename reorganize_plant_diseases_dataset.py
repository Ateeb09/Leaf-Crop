"""
Reorganize 'New Plant Diseases Dataset' (38 classes) into per-crop Train/Val/Test
so leaf_model_multi_crop.py can train 14 crops: Apple, Blueberry, Cherry, Corn, Grape,
Orange, Peach, Pepper, Potato, Raspberry, Soybean, Squash, Strawberry, Tomato.

Usage:
  1. Download from: https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset
  2. Unzip so you have a folder with 'train' and 'valid' (and optionally 'test').
  3. Set SOURCE_DIR below to that folder, set OUT_DIR to where you want the new structure.
  4. Run: python reorganize_plant_diseases_dataset.py
  5. Add OUT_DIR to DATA_ROOTS in leaf_model_multi_crop.py (or copy into archive_extra_crops).
"""
import os
import shutil
from collections import defaultdict

# Path to the unzipped "New Plant Diseases Dataset" (contains train, valid, test)
SOURCE_DIR = r"C:\Users\Ateeb Ahmad\Downloads\archive5"
# Output folder: will create CropName/Train, CropName/Val, CropName/Test
OUT_DIR = os.path.join(os.path.dirname(__file__), "archive_extra_crops")

# Class names in that dataset are like "Tomato___Bacterial_spot" -> crop = "Tomato"
def crop_from_class(class_name):
    if "___" in class_name:
        return class_name.split("___")[0].strip()
    return class_name


def main():
    if not os.path.isdir(SOURCE_DIR):
        print(f"SOURCE_DIR not found: {SOURCE_DIR}")
        print("Download the dataset and set SOURCE_DIR in this script.")
        return
    os.makedirs(OUT_DIR, exist_ok=True)
    for split in ["train", "valid"]:
        src = os.path.join(SOURCE_DIR, split)
        if not os.path.isdir(src):
            print(f"Skipping (not found): {src}")
            continue
        for class_name in os.listdir(src):
            crop = crop_from_class(class_name)
            if not crop:
                continue
            # e.g. archive_extra_crops/Tomato/Train/Tomato___Bacterial_spot/
            split_cap = "Val" if split == "valid" else "Train"
            dest_base = os.path.join(OUT_DIR, crop, split_cap, class_name)
            src_class = os.path.join(src, class_name)
            if not os.path.isdir(src_class):
                continue
            os.makedirs(dest_base, exist_ok=True)
            for f in os.listdir(src_class):
                src_f = os.path.join(src_class, f)
                if os.path.isfile(src_f) and f.lower().endswith((".jpg", ".jpeg", ".png")):
                    shutil.copy2(src_f, os.path.join(dest_base, f))
            print(f"Copied {src_class} -> {dest_base}")
    # Create Test by splitting 15% of Val (if no test folder in source)
    test_src = os.path.join(SOURCE_DIR, "test")
    if os.path.isdir(test_src):
        for class_name in os.listdir(test_src):
            crop = crop_from_class(class_name)
            if not crop:
                continue
            dest_base = os.path.join(OUT_DIR, crop, "Test", class_name)
            src_class = os.path.join(test_src, class_name)
            if not os.path.isdir(src_class):
                continue
            os.makedirs(dest_base, exist_ok=True)
            for f in os.listdir(src_class):
                src_f = os.path.join(src_class, f)
                if os.path.isfile(src_f) and f.lower().endswith((".jpg", ".jpeg", ".png")):
                    shutil.copy2(src_f, os.path.join(dest_base, f))
        print("Copied test set.")
    else:
        # No test folder: create Test by symlink/copy 15% of Val (simplified: copy first 5 images per class from Val)
        for crop in os.listdir(OUT_DIR):
            val_path = os.path.join(OUT_DIR, crop, "Val")
            test_path = os.path.join(OUT_DIR, crop, "Test")
            if not os.path.isdir(val_path):
                continue
            os.makedirs(test_path, exist_ok=True)
            for class_name in os.listdir(val_path):
                val_class = os.path.join(val_path, class_name)
                test_class = os.path.join(test_path, class_name)
                if not os.path.isdir(val_class):
                    continue
                os.makedirs(test_class, exist_ok=True)
                files = [f for f in os.listdir(val_class) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
                for f in files[: max(1, len(files) // 5)]:  # ~20% to test
                    shutil.copy2(os.path.join(val_class, f), os.path.join(test_class, f))
        print("Created Test from subset of Val (no test folder in source).")
    print(f"Done. Output: {OUT_DIR}")


if __name__ == "__main__":
    main()
