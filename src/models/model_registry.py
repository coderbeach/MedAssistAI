import torch
import torch.nn as nn
import torchvision.models as models
from transformers import AutoModelForImageClassification, AutoConfig

def load_resnet50(num_classes=13, dropout=0.3):
    # Load ResNet50 with default weights
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    
    # Replace classification head (fc)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, num_classes)
    )
    return model

def load_huggingface_model(model_id="microsoft/swin-tiny-patch4-window7-224", num_classes=13, dropout=0.2):
    # Load configuration
    config = AutoConfig.from_pretrained(model_id)
    config.num_labels = num_classes
    config.id2label = {str(i): c for c, i in {
        "acne": 0, "eczema": 1, "psoriasis": 2, "ringworm": 3, "vitiligo": 4,
        "chickenpox": 5, "measles": 6, "fungal_infection": 7, "dermatitis": 8,
        "suspicious_lesion": 9, "stye": 10, "conjunctivitis": 11, "normal_eye": 12
    }.items()}
    config.label2id = {c: i for i, c in config.id2label.items()}
    
    # Load model with classification head
    model = AutoModelForImageClassification.from_pretrained(
        model_id, 
        config=config,
        ignore_mismatched_sizes=True
    )
    return model

def get_model(model_name, num_classes=13, dropout=0.3, hf_model_id=None):
    if model_name.lower() == "resnet50":
        return load_resnet50(num_classes, dropout)
    elif "swin" in model_name.lower() or "vit" in model_name.lower():
        model_id = hf_model_id or "microsoft/swin-tiny-patch4-window7-224"
        return load_huggingface_model(model_id, num_classes, dropout)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

def freeze_or_unfreeze_layers(model, unfreeze_layers):
    """
    Freezes all layers of the model, and then unfreezes only the ones
    matching the strings in `unfreeze_layers`.
    If 'all' is in `unfreeze_layers`, all parameters are unfrozen.
    """
    if "all" in unfreeze_layers:
        print("  Unfreezing all layers...")
        for param in model.parameters():
            param.requires_grad = True
        return
        
    # First, freeze everything
    for param in model.parameters():
        param.requires_grad = False
        
    # Unfreeze specific layers by name
    unfrozen_params = 0
    for name, param in model.named_parameters():
        should_unfreeze = False
        for target in unfreeze_layers:
            if target in name:
                should_unfreeze = True
                break
                
        if should_unfreeze:
            param.requires_grad = True
            unfrozen_params += 1
            
    print(f"  Unfroze {unfrozen_params} parameter modules matching: {unfreeze_layers}")
