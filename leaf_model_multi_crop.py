"""
Train / validate / test leaf disease models for many crops using deep learning.
Supports multiple dataset roots: add 10-15+ more crops by placing another folder
with the same structure (CropName/Train, Val, Test with class subfolders).

Example: Download "New Plant Diseases Dataset" or "Plant Village" from Kaggle,
organize or merge so each crop has Train/Val/Test, then add that path to DATA_ROOTS.
"""
import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
from sklearn.metrics import classification_report, confusion_matrix

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 25  # Early stopping will stop earlier to avoid overfitting

# Primary dataset (your current 9 crops)
DATA_ROOTS = [
    r"C:\Users\Ateeb Ahmad\Downloads\archive (4)",
]
# Optional: add a second folder with 10-15 more crops (same structure: CropName/Train, Val, Test)
# Example: r"C:\Users\Ateeb Ahmad\Downloads\archive_extra_crops"
EXTRA_DATA_ROOT = os.path.join(os.path.dirname(__file__), "archive_extra_crops")
if os.path.isdir(EXTRA_DATA_ROOT):
    DATA_ROOTS.append(EXTRA_DATA_ROOT)


def discover_crop_folders(roots):
    """Return list of (root, crop_folder_name) for each folder that has Train, Val, Test."""
    crops = []
    for root in roots:
        if not os.path.isdir(root):
            continue
        for name in sorted(os.listdir(root)):
            path = os.path.join(root, name)
            if not os.path.isdir(path):
                continue
            if os.path.isdir(os.path.join(path, "Train")) and \
               os.path.isdir(os.path.join(path, "Val")) and \
               os.path.isdir(os.path.join(path, "Test")):
                crops.append((root, name))
    return crops


def safe_name(folder_name):
    return folder_name.lower().replace(" ", "_").replace("(", "").replace(")", "")


def main():
    all_crops = discover_crop_folders(DATA_ROOTS)
    if not all_crops:
        print("No crop folders (with Train/Val/Test) found in DATA_ROOTS:", DATA_ROOTS)
        return
    print(f"Found {len(all_crops)} crops: {[c[1] for c in all_crops]}")

    for data_root, crop_folder in all_crops:
        crop_name = crop_folder
        train_path = os.path.join(data_root, crop_folder, "Train")
        val_path = os.path.join(data_root, crop_folder, "Val")
        test_path = os.path.join(data_root, crop_folder, "Test")

        print("=" * 60)
        print(f"Training model for: {crop_name}")
        print("=" * 60)

        train_datagen = ImageDataGenerator(
            rescale=1.0 / 255,
            rotation_range=25,
            width_shift_range=0.15,
            height_shift_range=0.15,
            zoom_range=0.2,
            horizontal_flip=True,
            shear_range=0.1,
            brightness_range=[0.85, 1.15],
            fill_mode="nearest",
        )
        val_test_datagen = ImageDataGenerator(rescale=1.0 / 255)

        train_data = train_datagen.flow_from_directory(
            train_path,
            target_size=(IMG_SIZE, IMG_SIZE),
            batch_size=BATCH_SIZE,
            class_mode="categorical",
        )
        val_data = val_test_datagen.flow_from_directory(
            val_path,
            target_size=(IMG_SIZE, IMG_SIZE),
            batch_size=BATCH_SIZE,
            class_mode="categorical",
            classes=list(train_data.class_indices.keys()),
        )
        test_data = val_test_datagen.flow_from_directory(
            test_path,
            target_size=(IMG_SIZE, IMG_SIZE),
            batch_size=BATCH_SIZE,
            class_mode="categorical",
            classes=list(train_data.class_indices.keys()),
            shuffle=False,
        )

        base_model = MobileNetV2(
            weights="imagenet",
            include_top=False,
            input_shape=(IMG_SIZE, IMG_SIZE, 3),
        )
        base_model.trainable = False

        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(64, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
            layers.Dropout(0.5),
            layers.Dense(train_data.num_classes, activation="softmax"),
        ])
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        model.summary()

        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=4,
            restore_best_weights=True,
            verbose=1,
        )
        history = model.fit(
            train_data,
            validation_data=val_data,
            epochs=EPOCHS,
            callbacks=[early_stop],
        )

        # Validate
        val_loss, val_acc = model.evaluate(val_data)
        print(f"[{crop_name}] Validation loss: {val_loss:.4f}, accuracy: {val_acc:.4f}")

        # Test
        test_loss, test_acc = model.evaluate(test_data)
        print(f"[{crop_name}] Test loss: {test_loss:.4f}, accuracy: {test_acc:.4f}")

        # Test set classification report and confusion matrix
        test_data.reset()
        pred_proba = model.predict(test_data)
        pred_idx = np.argmax(pred_proba, axis=1)
        true_idx = test_data.classes
        class_names = list(train_data.class_indices.keys())
        print(f"[{crop_name}] Test classification report:")
        print(classification_report(true_idx, pred_idx, target_names=class_names, zero_division=0))
        print(f"[{crop_name}] Test confusion matrix (rows=true, cols=pred):")
        print(confusion_matrix(true_idx, pred_idx))

        # Save
        sn = safe_name(crop_folder)
        model_file = f"leaf_classifier_{sn}.h5"
        classes_file = f"leaf_classes_{sn}.json"
        model.save(model_file)
        with open(classes_file, "w") as f:
            json.dump(train_data.class_indices, f, indent=2)
        print(f"[{crop_name}] Saved {model_file}, {classes_file}")

    print("Training, validation, and testing complete for all crops.")


if __name__ == "__main__":
    main()
