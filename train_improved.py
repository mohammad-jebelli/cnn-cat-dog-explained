"""
train_improved.py
=================
Improved Cat/Dog CNN with BatchNorm + Dropout + padding.
GPU-aware: automatically uses the GPU if one is available.

Run:
    python train_improved.py

Dataset layout (same as the original README):
    Dogs-vs-Cats/
    |-- training_set/training_set/{cats,dogs}/
    +-- test_set/test_set/{cats,dogs}/

Outputs:
    cat_dog_improved.keras   <- trained model (modern Keras format)
    training_history.png     <- accuracy / loss curves
"""

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# 0) GPU check (informational)
# ---------------------------------------------------------------------------
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    # Grow memory gradually instead of grabbing all VRAM up front
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
    print(f"[OK] GPU found: {[g.name for g in gpus]} -- training on GPU")
else:
    print("[WARN] No GPU found -- training on CPU (slower)")

# ---------------------------------------------------------------------------
# 1) Data -- same paths and same light augmentation as the original
#    rescale=1/255 : input normalization (0..255 -> 0..1)
# ---------------------------------------------------------------------------
IMG_SIZE = (64, 64)
BATCH = 32

train_dir = "Dogs-vs-Cats/training_set/training_set"
test_dir = "Dogs-vs-Cats/test_set/test_set"

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
)
test_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_generator = train_datagen.flow_from_directory(
    train_dir, target_size=IMG_SIZE, batch_size=BATCH, class_mode="binary"
)
test_generator = test_datagen.flow_from_directory(
    test_dir, target_size=IMG_SIZE, batch_size=BATCH, class_mode="binary"
)

# Class mapping (which index = cat/dog) -- needed to interpret sigmoid output
print("Class mapping:", train_generator.class_indices)

# ---------------------------------------------------------------------------
# 2) Improved model
#    Differences vs the original (each maps to a concept):
#      * padding="same"       -> image does not shrink at each Conv (keeps edges)
#      * BatchNormalization    -> training stability and speed
#      * Dropout               -> less overfitting (active only during training)
#    Note: BatchNorm sits after Conv and before ReLU (common pattern),
#    so activation is separated from Conv and written explicitly.
# ---------------------------------------------------------------------------
model = models.Sequential([
    layers.Input(shape=(64, 64, 3)),

    # --- block 1 ---
    layers.Conv2D(32, (3, 3), padding="same"),
    layers.BatchNormalization(),
    layers.Activation("relu"),
    layers.MaxPooling2D(pool_size=(2, 2)),

    # --- block 2 ---
    layers.Conv2D(64, (3, 3), padding="same"),
    layers.BatchNormalization(),
    layers.Activation("relu"),
    layers.MaxPooling2D(pool_size=(2, 2)),

    # --- block 3 (one deeper block than the original) ---
    layers.Conv2D(128, (3, 3), padding="same"),
    layers.BatchNormalization(),
    layers.Activation("relu"),
    layers.MaxPooling2D(pool_size=(2, 2)),

    # --- classification ---
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.5),               # 50% of neurons dropped during training
    layers.Dense(1, activation="sigmoid"),
])

model.summary()

# ---------------------------------------------------------------------------
# 3) Compile and train
# ---------------------------------------------------------------------------
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

EPOCHS = 15

# Save the BEST model (highest val_accuracy) instead of just the last epoch.
# After training, cat_dog_improved.keras holds the best-performing weights.
checkpoint = ModelCheckpoint(
    "cat_dog_improved.keras",
    monitor="val_accuracy",   # watch validation accuracy
    save_best_only=True,      # overwrite only when it improves
    mode="max",               # higher is better
    verbose=1,                # print a line when a better model is saved
)

history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=test_generator,
    callbacks=[checkpoint],
)

# ---------------------------------------------------------------------------
# 4) The best model was already saved by ModelCheckpoint during training.
#    (No final model.save needed -- that would overwrite the best with the last.)
# ---------------------------------------------------------------------------
print("[OK] Best model saved during training: cat_dog_improved.keras")

# ---------------------------------------------------------------------------
# 5) Accuracy and loss curves
# ---------------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(history.history["accuracy"], label="train")
ax1.plot(history.history["val_accuracy"], label="validation")
ax1.set_title("Accuracy")
ax1.set_xlabel("Epoch")
ax1.legend()

ax2.plot(history.history["loss"], label="train")
ax2.plot(history.history["val_loss"], label="validation")
ax2.set_title("Loss")
ax2.set_xlabel("Epoch")
ax2.legend()

plt.tight_layout()
plt.savefig("training_history.png", dpi=120)
print("[OK] Curves saved: training_history.png")
