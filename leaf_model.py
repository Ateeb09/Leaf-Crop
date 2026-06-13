import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 20  # Early stopping will stop earlier to avoid overfitting

# Root of Kaggle Plant Village dataset
DATA_ROOT = r"C:\Users\Ateeb Ahmad\Downloads\archive (4)"

# Crop folders and their subfolder names
CROP_CONFIGS = {
    "Apple": "Apple",
    "Bell Pepper": "Bell Pepper",
    "Cherry": "Cherry",
    "Corn (Maize)": "Corn (Maize)",
    "Grape": "Grape",
    "Peach": "Peach",
    "Potato": "Potato",
    "Strawberry": "Strawberry",
    "Tomato": "Tomato",
}

import os
import json

for crop_name, crop_folder in CROP_CONFIGS.items():
    print("=" * 60)
    print(f"Training model for: {crop_name}")
    print("=" * 60)

    train_path = os.path.join(DATA_ROOT, crop_folder, "Train")
    val_path = os.path.join(DATA_ROOT, crop_folder, "Val")
    test_path = os.path.join(DATA_ROOT, crop_folder, "Test")

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

    val_test_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
    )

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
    )

    base_model = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
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

    val_loss, val_acc = model.evaluate(val_data)
    print(f"[{crop_name}] Validation loss: {val_loss:.4f}")
    print(f"[{crop_name}] Validation accuracy: {val_acc:.4f}")

    test_loss, test_acc = model.evaluate(test_data)
    print(f"[{crop_name}] Test loss: {test_loss:.4f}")
    print(f"[{crop_name}] Test accuracy: {test_acc:.4f}")

    # Save model and classes specific to this crop
    safe_name = crop_folder.lower().replace(" ", "_").replace("(", "").replace(")", "")
    model_file = f"leaf_classifier_{safe_name}.h5"
    classes_file = f"leaf_classes_{safe_name}.json"

    model.save(model_file)
    with open(classes_file, "w") as f:
        json.dump(train_data.class_indices, f)

    print(f"[{crop_name}] Saved model to {model_file}")
    print(f"[{crop_name}] Saved classes to {classes_file}")

print("Training, validation, and testing complete for all crops.")