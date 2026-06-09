import os
import urllib.request
import pandas as pd
import numpy as np
import pickle
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Ensure project directories exist
DATA_DIR = "./data"
MODEL_DIR = "./models"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# URLs for datasets on GitHub (Kaushil268 dataset mirror)
TRAIN_URL = "https://raw.githubusercontent.com/anujdutt9/Disease-Prediction-from-Symptoms/master/dataset/training_data.csv"
TEST_URL = "https://raw.githubusercontent.com/anujdutt9/Disease-Prediction-from-Symptoms/master/dataset/test_data.csv"

def download_data():
    train_path = os.path.join(DATA_DIR, "training.csv")
    test_path = os.path.join(DATA_DIR, "testing.csv")
    
    if not os.path.exists(train_path):
        print("Downloading training data...")
        urllib.request.urlretrieve(TRAIN_URL, train_path)
    if not os.path.exists(test_path):
        print("Downloading testing data...")
        urllib.request.urlretrieve(TEST_URL, test_path)
        
    return pd.read_csv(train_path), pd.read_csv(test_path)

def augment_with_extra_diseases(df, columns):
    """
    Synthetically appends 10 additional common diseases to reach the 51 disease threshold,
    using medical symptoms present in the 132 symptoms columns of the dataset.
    Differentiates between core (mandatory) and optional symptoms to prevent misdiagnosis.
    """
    print("Augmenting dataset with 10 additional common diseases using strict clinical rules...")
    
    extra_diseases = {
        "Appendicitis": {
            "core": ["abdominal_pain", "vomiting"],
            "optional": ["fever", "nausea"]
        },
        "COPD": {
            "core": ["breathlessness", "cough"],
            "optional": ["fatigue", "chest_pain"]
        },
        "COVID-19": {
            "core": ["fever", "cough", "loss_of_smell"],
            "optional": ["continuous_sneezing", "headache", "fatigue"]
        },
        "Influenza": {
            "core": ["high_fever", "chills", "muscle_weakness"],
            "optional": ["cough", "headache", "throat_irritation"]
        },
        "Kidney Stones": {
            "core": ["back_pain", "burning_micturition"],
            "optional": ["vomiting", "nausea"]
        },
        "Gastrointestinal Bleeding": {
            "core": ["stomach_bleeding"],
            "optional": ["vomiting", "fatigue", "dark_urine"]
        },
        "Iron Deficiency Anemia": {
            "core": ["lethargy", "fatigue"],
            "optional": ["breathlessness", "headache"]
        },
        "Transient Ischemic Attack (TIA)": {
            "core": ["weakness_of_one_body_side", "slurred_speech"],
            "optional": ["headache", "dizziness", "loss_of_balance"]
        },
        "Acute Gastritis": {
            "core": ["indigestion", "stomach_pain"],
            "optional": ["nausea", "vomiting"]
        },
        "Acute Sinusitis": {
            "core": ["sinus_pressure", "runny_nose"],
            "optional": ["headache", "continuous_sneezing"]
        }
    }
    
    new_rows = []
    # Generate 120 samples per new disease with variations in optional symptoms
    for disease, sym_config in extra_diseases.items():
        core_syms = sym_config["core"]
        opt_syms = sym_config["optional"]
        
        for _ in range(120):
            row = {col: 0 for col in columns}
            row['prognosis'] = disease
            
            # Enforce core symptoms (must be present)
            for sym in core_syms:
                if sym in row:
                    row[sym] = 1
                    
            # Add random subset of optional symptoms
            if opt_syms:
                num_opts = np.random.randint(0, len(opt_syms) + 1)
                selected_opts = np.random.choice(opt_syms, num_opts, replace=False)
                for sym in selected_opts:
                    if sym in row:
                        row[sym] = 1
                        
            new_rows.append(row)
            
    df_extra = pd.DataFrame(new_rows)
    df_augmented = pd.concat([df, df_extra], ignore_index=True)
    return df_augmented

# PyTorch Dataset Definition
class SymptomDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        
    def __len__(self):
        return len(self.y)
        
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# PyTorch MLP Neural Network (Day 3 Concept)
class SymptomMLP(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(SymptomMLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, num_classes)
        )
        
    def forward(self, x):
        return self.network(x)

def train():
    # Load dataset
    df_train, df_test = download_data()
    
    # Strip whitespace from column names if present
    df_train.columns = df_train.columns.str.strip()
    df_test.columns = df_test.columns.str.strip()
    
    # Get feature column list
    feature_cols = [c for c in df_train.columns if c != 'prognosis' and c != 'Unnamed: 133']
    
    # Augment both datasets
    df_train = augment_with_extra_diseases(df_train, feature_cols)
    df_test = augment_with_extra_diseases(df_test, feature_cols)
    
    # Encode target labels
    le = LabelEncoder()
    df_train['prognosis'] = df_train['prognosis'].str.strip()
    df_test['prognosis'] = df_test['prognosis'].str.strip()
    
    # Fit label encoder on combined classes
    all_classes = pd.concat([df_train['prognosis'], df_test['prognosis']]).unique()
    le.fit(all_classes)
    
    # Save label encoder class list for UI decoding
    classes_dict = {i: cls for i, cls in enumerate(le.classes_)}
    with open(os.path.join(MODEL_DIR, "symptom_classes.json"), "w") as f:
        json.dump(classes_dict, f, indent=4)
        
    # Save the feature column order for model input consistency
    with open(os.path.join(MODEL_DIR, "symptom_features.json"), "w") as f:
        json.dump(feature_cols, f, indent=4)
        
    X_train = df_train[feature_cols].values
    y_train = le.transform(df_train['prognosis'])
    
    X_test = df_test[feature_cols].values
    y_test = le.transform(df_test['prognosis'])
    
    # ----------------------------------------------------
    # ALGORITHM 1: Scikit-learn Random Forest (Day 2 ML Concept)
    # ----------------------------------------------------
    print("\n--- Training Random Forest Classifier ---")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    
    rf_preds = rf_model.predict(X_test)
    rf_acc = accuracy_score(y_test, rf_preds)
    print(f"Random Forest Test Accuracy: {rf_acc * 100:.2f}%")
    
    # Save Random Forest Model
    with open(os.path.join(MODEL_DIR, "symptom_rf.pkl"), "wb") as f:
        pickle.dump(rf_model, f)
    print("Saved Random Forest model to models/symptom_rf.pkl")
    
    # ----------------------------------------------------
    # ALGORITHM 2: PyTorch Multi-Layer Perceptron (Day 3 DL Concept)
    # ----------------------------------------------------
    print("\n--- Training PyTorch MLP Neural Network ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    input_dim = len(feature_cols)
    num_classes = len(le.classes_)
    
    mlp_model = SymptomMLP(input_dim, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(mlp_model.parameters(), lr=0.005, weight_decay=1e-4)
    
    train_dataset = SymptomDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    # Training Loop
    mlp_model.train()
    epochs = 30
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = mlp_model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss/len(train_loader):.4f}")
            
    # Evaluation
    mlp_model.eval()
    with torch.no_grad():
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
        y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)
        
        test_outputs = mlp_model(X_test_tensor)
        _, mlp_preds = torch.max(test_outputs, 1)
        mlp_acc = (mlp_preds == y_test_tensor).sum().item() / len(y_test)
        print(f"PyTorch MLP Test Accuracy: {mlp_acc * 100:.2f}%")
        
    # Save PyTorch Model
    torch.save(mlp_model.state_dict(), os.path.join(MODEL_DIR, "symptom_mlp.pth"))
    print("Saved PyTorch MLP model to models/symptom_mlp.pth")
    print("\nSymptom Classifier Training Finished Successfully!")

if __name__ == "__main__":
    train()
