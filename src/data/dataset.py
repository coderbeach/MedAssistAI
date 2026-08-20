import os
from PIL import Image
from torch.utils.data import Dataset

# Standard class mapping
CLASS_MAP = {
    "acne": 0,
    "eczema": 1,
    "psoriasis": 2,
    "ringworm": 3,
    "vitiligo": 4,
    "chickenpox": 5,
    "measles": 6,
    "fungal_infection": 7,
    "dermatitis": 8,
    "suspicious_lesion": 9,
    "stye": 10,
    "conjunctivitis": 11,
    "normal_eye": 12
}

class MedicalDataset(Dataset):
    def __init__(self, split_dir, transform=None):
        self.split_dir = split_dir
        self.transform = transform
        self.samples = []
        
        if not os.path.exists(split_dir):
            return
            
        for cls_name, cls_idx in CLASS_MAP.items():
            cls_folder = os.path.join(split_dir, cls_name)
            if os.path.isdir(cls_folder):
                for f in os.listdir(cls_folder):
                    if f.lower().endswith((".jpg", ".jpeg", ".png")):
                        img_path = os.path.join(cls_folder, f)
                        self.samples.append((img_path, cls_idx))
                        
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            with Image.open(img_path) as img:
                img = img.convert("RGB")
                if self.transform:
                    img = self.transform(img)
                return img, label
        except Exception as e:
            # Fallback if image load failed
            print(f"Error loading image {img_path}: {e}")
            # Try loading the next index
            next_idx = (idx + 1) % len(self.samples)
            return self.__getitem__(next_idx)
