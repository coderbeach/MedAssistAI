import os

DATA_DIR = "./data/external_images"

def clean_synthetic_files():
    print("Cleaning up old synthetic/mock images from dataset...")
    removed_count = 0
    kept_count = 0
    
    for split in ["train", "val"]:
        split_dir = os.path.join(DATA_DIR, split)
        if not os.path.exists(split_dir):
            continue
            
        for cls in os.listdir(split_dir):
            cls_dir = os.path.join(split_dir, cls)
            if not os.path.isdir(cls_dir):
                continue
                
            for filename in os.listdir(cls_dir):
                filepath = os.path.join(cls_dir, filename)
                if os.path.isfile(filepath):
                    if not filename.startswith("real_"):
                        # Delete old synthetic/duplicate file
                        try:
                            os.remove(filepath)
                            removed_count += 1
                        except Exception as e:
                            print(f"Failed to delete {filepath}: {e}")
                    else:
                        kept_count += 1
                        
    print(f"Cleanup complete! Removed {removed_count} synthetic files. Kept {kept_count} real files.")

if __name__ == "__main__":
    clean_synthetic_files()
