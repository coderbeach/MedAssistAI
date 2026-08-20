import torchvision.transforms as T

def get_transforms(image_size=224, aug_strength="moderate"):
    # ImageNet normalization stats
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    # Deterministic transforms for validation and testing
    val_test_transform = T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std)
    ])
    
    # Augmented transforms for training
    if aug_strength == "none":
        train_transform = val_test_transform
    elif aug_strength == "light":
        train_transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(degrees=10),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std)
        ])
    else: # moderate (default for medical images)
        train_transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.2),
            T.RandomRotation(degrees=15),
            T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
            T.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std)
        ])
        
    return train_transform, val_test_transform
