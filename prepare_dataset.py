import os
import shutil
import random

# Configuration
SOURCE_DIR = "data2"  # Your copied Assignment 1 folder structure
TARGET_DIR = "data"   # New Assignment 2 dataset folder
CLASSES = ["peace", "stop", "thumbs_up"]
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
RANDOM_SEED = 42

random.seed(RANDOM_SEED)

def setup_directories():
    for split in ["train", "val", "test"]:
        for cls in CLASSES:
            os.makedirs(os.path.join(TARGET_DIR, split, cls), exist_ok=True)

def process_dataset():
    setup_directories()
    
    for cls in CLASSES:
        # Collect all images from both train and test folders in data2
        all_images = []
        for set_type in ["train", "test"]:
            class_path = os.path.join(SOURCE_DIR, set_type, cls)
            if os.path.exists(class_path):
                images = [os.path.join(class_path, img) for img in os.listdir(class_path) 
                          if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
                all_images.extend(images)
        
        # Shuffle images randomly to mix white and gray backgrounds across splits
        random.shuffle(all_images)
        
        total_images = len(all_images)
        val_count = int(total_images * VAL_SPLIT)
        test_count = int(total_images * TEST_SPLIT)
        
        val_imgs = all_images[:val_count]
        test_imgs = all_images[val_count:val_count + test_count]
        train_imgs = all_images[val_count + test_count:]
        
        # Copy files to target directories
        for img_path in train_imgs:
            shutil.copy(img_path, os.path.join(TARGET_DIR, "train", cls, os.path.basename(img_path)))
        for img_path in val_imgs:
            shutil.copy(img_path, os.path.join(TARGET_DIR, "val", cls, os.path.basename(img_path)))
        for img_path in test_imgs:
            shutil.copy(img_path, os.path.join(TARGET_DIR, "test", cls, os.path.basename(img_path)))
            
        print(f"Class '{cls}': Total={total_images} | Train={len(train_imgs)} | Val={len(val_imgs)} | Test={len(test_imgs)}")

if __name__ == "__main__":
    process_dataset()
    print("\nDataset successfully restructured into 'data/' for Assignment 2!")