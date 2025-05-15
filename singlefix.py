import os
import tensorflow as tf

image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')
dataset_path = "dataset"

print("🧪 Testing image files one by one...")

bad_files = []

for root, _, files in os.walk(dataset_path):
    for file in files:
        if file.lower().endswith(image_extensions):
            path = os.path.join(root, file)
            try:
                img_data = tf.io.read_file(path)
                _ = tf.image.decode_image(img_data, channels=3)
            except Exception as e:
                print(f"❌ Failed to decode: {path}\n   ↪ Error: {e}")
                bad_files.append(path)

print(f"\n✅ Done checking. Bad files found: {len(bad_files)}")

if bad_files:
    print("\nList of corrupt/unreadable images:")
    for file in bad_files:
        print(f" - {file}")
else:
    print("🎉 All image files are readable by TensorFlow.")
