import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import numpy as np
import json
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Directory configurations
DATA_DIR = "./data/external_images"
MODEL_DIR = "./models"
os.makedirs(MODEL_DIR, exist_ok=True)

# 13 Target classes (skin and external conditions, including eye conditions)
CLASSES = [
    "Acne", "Eczema", "Psoriasis", "Ringworm", "Vitiligo", 
    "Chickenpox rash", "Measles rash", "Fungal infection", "Dermatitis", 
    "Suspicious skin lesion", "Stye", "Conjunctivitis", "Normal Eye"
]

class ExternalImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        for label_idx, cls in enumerate(CLASSES):
            cls_folder = os.path.join(root_dir, cls.replace(" ", "_"))
            if os.path.exists(cls_folder):
                for file_name in os.listdir(cls_folder):
                    if file_name.endswith((".jpg", ".png", ".jpeg")):
                        self.image_paths.append(os.path.join(cls_folder, file_name))
                        self.labels.append(label_idx)
                        
    def __len__(self):
        return len(self.image_paths)
        
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            # Fallback to black image if image load fails during training
            print(f"Error loading {img_path}: {e}. Loading black placeholder.")
            image = Image.new("RGB", (224, 224), color=(0, 0, 0))
            
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

def train_cnn():
    # Save target classes mapping to JSON
    classes_dict = {i: name for i, name in enumerate(CLASSES)}
    with open(os.path.join(MODEL_DIR, "image_classes.json"), "w") as f:
        json.dump(classes_dict, f, indent=4)
        
    train_dir = os.path.join(DATA_DIR, "train")
    if not os.path.exists(train_dir) or len(os.listdir(train_dir)) == 0:
        raise FileNotFoundError("Real images dataset not found. Please run download_real_images_v2.py first.")
        
    # Transformations (Data Augmentation for generalizing on real images)
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    train_dataset = ExternalImageDataset(os.path.join(DATA_DIR, "train"), transform=train_transform)
    val_dataset = ExternalImageDataset(os.path.join(DATA_DIR, "val"), transform=val_transform)
    
    print(f"Dataset Details: {len(train_dataset)} training images, {len(val_dataset)} validation images.")
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)
    
    # Setup Device & Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device for CNN training: {device}")
    
    # Load pre-trained ResNet-18 model
    try:
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    except AttributeError:
        model = models.resnet18(pretrained=True)
        
    # Freeze convolutional layers to allow fast CPU-based fine-tuning
    for param in model.parameters():
        param.requires_grad = False
        
    # Replace classification head
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(CLASSES))
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=0.005, weight_decay=1e-4)
    
    epochs = 3
    print("\nStarting CNN Transfer Learning Training on 13 external classes...")
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        corrects = 0
        total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
                
            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            corrects += torch.sum(preds == labels.data)
            total += labels.size(0)
            
        epoch_loss = running_loss / total
        epoch_acc = corrects.double() / total
        
        # Validation epoch loss
        model.eval()
        val_loss = 0.0
        val_corrects = 0
        val_total = 0
        with torch.no_grad():
            for val_images, val_labels in val_loader:
                val_images, val_labels = val_images.to(device), val_labels.to(device)
                val_outputs = model(val_images)
                loss = criterion(val_outputs, val_labels)
                val_loss += loss.item() * val_images.size(0)
                _, val_preds = torch.max(val_outputs, 1)
                val_corrects += torch.sum(val_preds == val_labels.data)
                val_total += val_labels.size(0)
                
        val_epoch_loss = val_loss / val_total if val_total > 0 else 0.0
        val_acc = val_corrects.double() / val_total if val_total > 0 else 0.0
        
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {epoch_loss:.4f} Acc: {epoch_acc*100:.2f}% | Val Loss: {val_epoch_loss:.4f} Acc: {val_acc*100:.2f}%")
        
    # Final Comprehensive Evaluation with Precision, Recall, F1-Score
    print("\nEvaluating final validation set details...")
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for val_images, val_labels in val_loader:
            val_images = val_images.to(device)
            val_outputs = model(val_images)
            _, val_preds = torch.max(val_outputs, 1)
            
            all_preds.extend(val_preds.cpu().numpy())
            all_targets.extend(val_labels.numpy())
            
    print("\n================== Classification Report (ResNet-18) ==================")
    print(classification_report(all_targets, all_preds, target_names=CLASSES, zero_division=0))
    print("=======================================================================")
    
    torch.save(model.state_dict(), os.path.join(MODEL_DIR, "skin_classifier.pth"))
    print("Saved PyTorch CNN model weights to models/skin_classifier.pth")
    print("CNN Image Classifier Training Completed Successfully!")

if __name__ == "__main__":
    train_cnn()
