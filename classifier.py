import tensorflow as tf
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os

# === CONFIG ===
data_dir = "dataset"
img_size = (224, 224)
batch_size = 16
model_path = "my_image_classifier.keras"

# === 1. Load Datasets ===
train_ds = image_dataset_from_directory(
    data_dir,
    image_size=img_size,
    batch_size=batch_size,
    validation_split=0.2,
    subset="training",
    seed=123
)

val_ds = image_dataset_from_directory(
    data_dir,
    image_size=img_size,
    batch_size=batch_size,
    validation_split=0.2,
    subset="validation",
    seed=123
)

class_names = train_ds.class_names
print("Classes:", class_names)

# === 2. Prefetch to speed up ===
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

# === 3. Data Augmentation ===
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])

# === 4. Build Model ===
base_model = MobileNetV2(input_shape=img_size + (3,), include_top=False, weights='imagenet')
base_model.trainable = False  # Initial freeze

model = models.Sequential([
    data_augmentation,
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(len(class_names), activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# === 5. Callbacks ===
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
checkpoint = ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True)

# === 6. Train Initial (frozen) ===
print("🧠 Training base model...")
model.fit(train_ds, validation_data=val_ds, epochs=30, callbacks=[early_stop, checkpoint])

# === 7. Fine-tuning ===
print("🔧 Fine-tuning top layers...")
base_model.trainable = True
for layer in base_model.layers[:100]:  # Freeze early layers
    layer.trainable = False

model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),  # Lower LR
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_ds, validation_data=val_ds, epochs=30, callbacks=[early_stop, checkpoint])

# === 8. Final Save (redundant if checkpoint used) ===
model.save(model_path)
print("✅ Model training complete and saved as", model_path)
