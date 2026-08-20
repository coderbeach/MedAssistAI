import os
import sys
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np
from collections import Counter
from src.data.dataset import MedicalDataset, CLASS_MAP
from src.preprocessing.transforms import get_transforms
from src.models.model_registry import get_model, freeze_or_unfreeze_layers

def compute_class_weights(dataset):
    labels = [sample[1] for sample in dataset.samples]
    total_samples = len(labels)
    class_counts = Counter(labels)
    
    # Calculate weights: total_samples / (num_classes * count)
    weights = []
    for i in range(13):
        count = class_counts.get(i, 0)
        if count > 0:
            weights.append(total_samples / (13.0 * count))
        else:
            weights.append(1.0) # Fallback for unrepresented classes
            
    return torch.tensor(weights, dtype=torch.float32)

def train_model(config_path, train_dir, val_dir, output_dir, device="cpu"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    print(f"\n=========================================================")
    print(f" Starting Training Loop for: {config['model_name']} on {device}")
    print(f"=========================================================")
    
    # Set seeds
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    
    # 1. Transforms
    train_transform, val_transform = get_transforms(
        image_size=config["image_size"],
        aug_strength=config.get("aug_strength", "moderate")
    )
    
    # 2. Datasets
    train_dataset = MedicalDataset(train_dir, transform=train_transform)
    val_dataset = MedicalDataset(val_dir, transform=val_transform)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    if len(train_dataset) == 0:
        print("Error: No training images found. Verify dataset directory.")
        return
        
    # 3. Handle Imbalance: Compute class weights for loss function
    class_weights = compute_class_weights(train_dataset).to(device)
    print(f"Calculated class weights (inverted frequency): {class_weights.tolist()}")
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config["batch_size"], 
        shuffle=True,
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config["batch_size"], 
        shuffle=False,
        num_workers=0
    )
    
    # 4. Model Setup
    model = get_model(
        model_name=config["model_name"],
        num_classes=config["num_classes"],
        dropout=config.get("dropout", 0.3),
        hf_model_id=config.get("hf_model_id")
    )
    model = model.to(device)
    
    best_val_loss = float("inf")
    best_val_acc = 0.0
    
    # 5. Staged Training
    stages = ["stage1", "stage2", "stage3"]
    
    for stage_name in stages:
        if stage_name not in config:
            continue
            
        stage_cfg = config[stage_name]
        epochs = stage_cfg["epochs"]
        lr = stage_cfg["lr"]
        unfreeze_targets = stage_cfg["unfreeze_layers"]
        
        print(f"\n--- Entering {stage_name.upper()} (Epochs: {epochs}, LR: {lr}) ---")
        
        # Unfreeze appropriate layers
        freeze_or_unfreeze_layers(model, unfreeze_targets)
        
        # Optimizer (only for parameters that require gradients)
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=config.get("weight_decay", 0.01))
        
        for epoch in range(epochs):
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                
                optimizer.zero_grad()
                
                # Check model output format (Hugging Face outputs a sequence-like dict)
                outputs = model(images)
                if hasattr(outputs, "logits"):
                    logits = outputs.logits
                else:
                    logits = outputs
                    
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * images.size(0)
                _, preds = torch.max(logits, 1)
                train_correct += torch.sum(preds == labels.data)
                train_total += images.size(0)
                
            train_loss = train_loss / train_total
            train_acc = train_correct.double() / train_total
            
            # Validation Step
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    if hasattr(outputs, "logits"):
                        logits = outputs.logits
                    else:
                        logits = outputs
                        
                    loss = criterion(logits, labels)
                    val_loss += loss.item() * images.size(0)
                    _, preds = torch.max(logits, 1)
                    val_correct += torch.sum(preds == labels.data)
                    val_total += images.size(0)
                    
            val_loss = val_loss / val_total
            val_acc = val_correct.double() / val_total
            
            print(f"  Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")
            
            # Checkpointing
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_acc = val_acc
                
                # Save best checkpoint
                os.makedirs(output_dir, exist_ok=True)
                save_path = os.path.join(output_dir, "best_model.pth")
                
                # Save state dict and model configs
                save_dict = {
                    "model_state_dict": model.state_dict(),
                    "model_name": config["model_name"],
                    "config": config,
                    "class_map": CLASS_MAP,
                    "val_loss": val_loss,
                    "val_acc": val_acc
                }
                torch.save(save_dict, save_path)
                # print(f"    [Checkpoint Saved] Best Validation Loss improved to {best_val_loss:.4f}")
                
    print(f"\nTraining complete! Best Validation Accuracy: {best_val_acc:.4f} | Best Val Loss: {best_val_loss:.4f}")
