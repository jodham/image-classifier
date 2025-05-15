import tensorflow as tf
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
from keras.models import load_model
import os

# Constants
data_dir = "dataset"
img_size = (224, 224)
batch_size = 16
model_path = "my_image_classifier.keras"
RETRAIN = True  # Set to True if you want to force retraining

# Load dataset once (used if retraining)
def prepare_datasets():
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
    return train_ds, val_ds

# Build model
def build_model(num_classes):
    base_model = MobileNetV2(input_shape=img_size + (3,), include_top=False, weights="imagenet")
    base_model.trainable = False  # Freeze base
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Load or train model
if os.path.exists(model_path) and not RETRAIN:
    print("✅ Loading existing model...")
    model = load_model(model_path)
    class_names = ['devotional', 'landscape', 'nature', 'people', 'flowers', 'animals']  # Adjust to match dataset
else:
    print("🧠 Training new model...")
    train_ds, val_ds = prepare_datasets()
    class_names = train_ds.class_names
    model = build_model(len(class_names))
    history = model.fit(train_ds, validation_data=val_ds, epochs=10)
    model.save(model_path)

# Image classification function
def classify_image(img_path):
    img = tf.keras.utils.load_img(img_path, target_size=img_size)
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    print(f"📸 This image most likely belongs to '{class_names[tf.argmax(score)]}' with {100 * tf.reduce_max(score):.2f}% confidence.")

# Example usage
classify_image("test/biometric.jpg")
