import os
import sys
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
import numpy as np

PROJECT_DIR = r"c:\Users\Nisarga N\OneDrive\Documents\My Projects\Healthcare-AI-Capstone"
sys.path.append(PROJECT_DIR)

from src.models.model_registry import get_model
from src.data.dataset import CLASS_MAP

IDX_TO_CLASS = {v: k for k, v in CLASS_MAP.items()}

def predict_image(image_path, checkpoint_path, device="cpu"):
    # 1. Load Checkpoint
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint file not found at {checkpoint_path}")
        return
        
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint["config"]
    
    # 2. Recreate and Load Model
    model = get_model(
        model_name=checkpoint["model_name"],
        num_classes=config["num_classes"],
        hf_model_id=config.get("hf_model_id")
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    
    # 3. Load Temperature Scaling Parameter
    temperature = checkpoint.get("temperature", 1.0)
    
    # 4. Standard Preprocessing
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    transform = T.Compose([
        T.Resize((config["image_size"], config["image_size"])),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std)
    ])
    
    try:
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            img_tensor = transform(img).unsqueeze(0).to(device)
    except Exception as e:
        print(f"Error reading image: {e}")
        return
        
    # 5. Model Inference
    with torch.no_grad():
        outputs = model(img_tensor)
        if hasattr(outputs, "logits"):
            logits = outputs.logits
        else:
            logits = outputs
            
        # Apply Temperature Scaling (Logits / T)
        calibrated_logits = logits / temperature
        probs = torch.softmax(calibrated_logits, dim=1).cpu().numpy()[0]
        
    # 6. Retrieve Top-3 Predictions
    top_indices = np.argsort(probs)[::-1][:3]
    
    print("\n" + "=" * 50)
    print("      MEDIASSIST AI CLINICAL IMAGE CLASSIFIER")
    print("=" * 50)
    print(f"Target Image: {os.path.basename(image_path)}")
    print(f"Calibration Temperature: T = {temperature:.4f}\n")
    
    print("## Calibrated Model Classifications:")
    for idx, class_idx in enumerate(top_indices):
        class_name = IDX_TO_CLASS[class_idx]
        confidence = probs[class_idx] * 100
        prefix = "⭐️ [TOP] " if idx == 0 else f"    [{idx+1}] "
        print(f"{prefix}{class_name:<25} : {confidence:.2f}%")
        
    print("\n" + "-" * 50)
    print("⚠️ RESEARCH DISCLAIMER & WARNING:")
    print("This is an experimental computer vision image classification tool.")
    print("It is NOT a medical diagnosis device and should NOT replace")
    print("professional clinical examination by a licensed medical practitioner.")
    print("=" * 50 + "\n")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/predict.py <image_path> [checkpoint_path]")
        sys.exit(1)
        
    img_path = sys.argv[1]
    ckpt_path = sys.argv[2] if len(sys.argv) > 2 else os.path.join(PROJECT_DIR, "models", "resnet50_run", "best_model.pth")
    
    predict_image(img_path, ckpt_path)
