import os
import random
import shutil

# Paths relative to project root (works from repo root or from scripts/)
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
RAW_DIR = os.path.join(PROJECT_ROOT, "dataset", "raw")
LABELS_DIR = os.path.join(PROJECT_ROOT, "dataset", "labels")

IMG_TRAIN = os.path.join(PROJECT_ROOT, "dataset", "images", "train")
IMG_VAL = os.path.join(PROJECT_ROOT, "dataset", "images", "val")
LBL_TRAIN = os.path.join(PROJECT_ROOT, "dataset", "labels", "train")
LBL_VAL = os.path.join(PROJECT_ROOT, "dataset", "labels", "val")

# Create folders
for path in [IMG_TRAIN, IMG_VAL, LBL_TRAIN, LBL_VAL]:
    os.makedirs(path, exist_ok=True)

# Get all images
images = [f for f in os.listdir(RAW_DIR) if f.endswith(".jpg")]

print(f"Total images: {len(images)}")

# Shuffle
random.shuffle(images)

# Split ratio
split_ratio = 0.8
split_index = int(len(images) * split_ratio)

train_images = images[:split_index]
val_images = images[split_index:]

print(f"Train: {len(train_images)} | Val: {len(val_images)}")

# Function to move files
def move_files(image_list, img_dest, lbl_dest):
    for img in image_list:
        label = img.replace(".jpg", ".txt")

        img_src = os.path.join(RAW_DIR, img)
        lbl_src = os.path.join(LABELS_DIR, label)

        # Only move if label exists
        if os.path.exists(lbl_src):
            shutil.copy(img_src, os.path.join(img_dest, img))
            shutil.copy(lbl_src, os.path.join(lbl_dest, label))

# Move files
move_files(train_images, IMG_TRAIN, LBL_TRAIN)
move_files(val_images, IMG_VAL, LBL_VAL)

print("Dataset split completed.")