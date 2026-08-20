import os
import sys
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, balanced_accuracy_score
from torch.utils.data import DataLoader
from src.data.dataset import MedicalDataset, CLASS_MAP
from src.preprocessing.transforms import get_transforms
from src.models.model_registry import get_model

# Reverse class mapping for label rendering
IDX_TO_CLASS = {v: k for k, v in CLASS_MAP.items()}

class TemperatureScaler(nn.Module):
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)
        
    def forward(self, logits):
        return logits / self.temperature
        
    def fit(self, val_logits, val_labels):
        # We optimize the temperature T using LBFGS to minimize validation NLL
        optimizer = torch.optim.LBFGS([self.temperature], lr=0.01, max_iter=50)
        
        # Cross entropy loss represents negative log likelihood
        nll_criterion = nn.CrossEntropyLoss()
        
        def eval_loss():
            optimizer.zero_grad()
            loss = nll_criterion(self.forward(val_logits), val_labels)
            loss.backward()
            return loss
            
        optimizer.step(eval_loss)
        # Ensure T is positive
        with torch.no_grad():
            self.temperature.clamp_(min=0.01)
            
        print(f"  [Temperature Scaling] Optimal Temperature calibrated to T = {self.temperature.item():.4f}")
        return self.temperature.item()

def collect_logits_and_labels(model, dataloader, device):
    model.eval()
    all_logits = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            outputs = model(images)
            if hasattr(outputs, "logits"):
                logits = outputs.logits
            else:
                logits = outputs
                
            all_logits.append(logits.cpu())
            all_labels.append(labels)
            
    return torch.cat(all_logits, dim=0), torch.cat(all_labels, dim=0)

def evaluate_model(checkpoint_path, val_dir, test_dir, results_dir, device="cpu"):
    print(f"\n=========================================================")
    print(f" Running Evaluation & Temperature Calibration...")
    print(f"=========================================================")
    
    # 1. Load Checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint["config"]
    
    model = get_model(
        model_name=checkpoint["model_name"],
        num_classes=config["num_classes"],
        hf_model_id=config.get("hf_model_id")
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    
    # 2. Setup Dataloaders
    _, val_transform = get_transforms(image_size=config["image_size"])
    val_dataset = MedicalDataset(val_dir, transform=val_transform)
    test_dataset = MedicalDataset(test_dir, transform=val_transform)
    
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)
    
    if len(val_dataset) == 0 or len(test_dataset) == 0:
        print("Error: Validation or Test datasets are empty. Verify splits.")
        return
        
    # 3. Fit Temperature Scaler on Validation Logits
    val_logits, val_labels = collect_logits_and_labels(model, val_loader, device)
    scaler = TemperatureScaler().to(device)
    scaler.fit(val_logits.to(device), val_labels.to(device))
    
    # Save optimal temperature back to the checkpoint
    checkpoint["temperature"] = scaler.temperature.item()
    torch.save(checkpoint, checkpoint_path)
    
    # 4. Evaluate on Test Dataset
    test_logits, test_labels = collect_logits_and_labels(model, test_loader, device)
    
    # Compute uncalibrated and calibrated predictions
    uncalibrated_probs = torch.softmax(test_logits, dim=1).numpy()
    calibrated_probs = torch.softmax(scaler(test_logits.to(device)).cpu(), dim=1).detach().numpy()
    
    preds = np.argmax(calibrated_probs, axis=1)
    targets = test_labels.numpy()
    
    # 5. Calculate Metrics
    acc = accuracy_score(targets, preds)
    bal_acc = balanced_accuracy_score(targets, preds)
    
    print("\n## Evaluation Summary on Held-Out Test Set")
    print(f"Accuracy: {acc:.4f}")
    print(f"Balanced Accuracy: {bal_acc:.4f}")
    
    # Save report
    os.makedirs(results_dir, exist_ok=True)
    report_path = os.path.join(results_dir, "test_evaluation_report.txt")
    
    report_lines = []
    report_lines.append("=========================================================")
    report_lines.append("        TEST EVALUATION & CALIBRATION REPORT")
    report_lines.append("=========================================================\n")
    report_lines.append(f"Model Name: {checkpoint['model_name']}")
    report_lines.append(f"Calibrated Temperature: T = {scaler.temperature.item():.4f}\n")
    report_lines.append(f"Accuracy: {acc:.4f}")
    report_lines.append(f"Balanced Accuracy: {bal_acc:.4f}\n")
    
    # Per-Class Metrics
    target_names = [IDX_TO_CLASS[i] for i in range(13) if i in np.unique(targets)]
    present_indices = [i for i in range(13) if i in np.unique(targets)]
    
    cls_report = classification_report(
        targets, 
        preds, 
        labels=present_indices,
        target_names=target_names,
        output_dict=True
    )
    
    report_lines.append("## Per-Class Metrics Table")
    report_lines.append(f"{'Class Name':<25} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Support':<10}")
    report_lines.append("-" * 75)
    for name in target_names:
        metrics = cls_report[name]
        report_lines.append(f"{name:<25} | {metrics['precision']:<10.4f} | {metrics['recall']:<10.4f} | {metrics['f1-score']:<10.4f} | {int(metrics['support']):<10}")
    report_lines.append("-" * 75)
    
    macro_avg = cls_report["macro avg"]
    report_lines.append(f"{'Macro Average':<25} | {macro_avg['precision']:<10.4f} | {macro_avg['recall']:<10.4f} | {macro_avg['f1-score']:<10.4f} | {int(macro_avg['support']):<10}\n")
    
    # Confusion Matrix
    cm = confusion_matrix(targets, preds, labels=present_indices)
    cm_norm = confusion_matrix(targets, preds, labels=present_indices, normalize="true")
    
    report_lines.append("## Normalized Confusion Matrix")
    # Header
    header_str = f"{'':<25} | " + " | ".join([f"{name[:10]:<10}" for name in target_names])
    report_lines.append(header_str)
    report_lines.append("-" * len(header_str))
    for r_idx, name in enumerate(target_names):
        row_str = f"{name:<25} | " + " | ".join([f"{cm_norm[r_idx, c_idx]:<10.2%}" for c_idx in range(len(target_names))])
        report_lines.append(row_str)
        
    report_str = "\n".join(report_lines)
    print(report_str)
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_str)
        
    print(f"\nEvaluation metrics report saved to: {report_path}")
