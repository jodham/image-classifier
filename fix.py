import os
from PIL import Image

supported_exts = (".jpg", ".jpeg", ".png", ".bmp", ".gif")
dataset_path = "dataset"

def is_valid_image(path):
    try:
        with Image.open(path) as img:
            img.verify()  # Validate structure
        return True
    except:
        return False

for root, _, files in os.walk(dataset_path):
    for file in files:
        path = os.path.join(root, file)
        ext = file.lower().endswith(supported_exts)
        if not ext or not is_valid_image(path):
            print(f"🗑️ Removing: {path}")
            os.remove(path)

print("✅ Cleanup complete! Try training again.")
