import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from collections import Counter
import matplotlib.pyplot as plt

NUMPY_OUTPUT_PATH = 'dataset'

class ISLDataset(Dataset):
    def __init__(self, X, y, augment=False):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
        self.augment = augment
        
    def __len__(self):
        return len(self.X)
        
    def __getitem__(self, idx):
        x_data = self.X[idx].clone()
        if self.augment:
            # 1. Scale Jitter (±10%)
            scale_factor = 1.0 + (torch.rand(1).item() * 0.2 - 0.1)
            x_data = x_data * scale_factor
            
            # 2. Random Rotation (±15 degrees around Y-axis)
            angle = (torch.rand(1).item() * 30 - 15) * (np.pi / 180.0)
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            rot_matrix = torch.tensor([
                [cos_a, 0, sin_a],
                [0, 1, 0],
                [-sin_a, 0, cos_a]
            ], dtype=torch.float32)

            # Features are 546: 273 positions + 273 velocities (91 3D points each)
            pos = x_data[:, :273].view(-1, 91, 3)
            pos = torch.matmul(pos, rot_matrix)
            x_data[:, :273] = pos.view(-1, 273)
            
            vel = x_data[:, 273:].view(-1, 91, 3)
            vel = torch.matmul(vel, rot_matrix)
            x_data[:, 273:] = vel.view(-1, 273)

            # 3. Gaussian Noise (±0.005)
            noise = torch.randn_like(x_data) * 0.005
            x_data = x_data + noise
            
        return x_data, self.y[idx]


class ASLModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        # Single LSTM layer to prevent overfitting
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True, num_layers=1)
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(hidden_size, 32)
        self.fc2 = nn.Linear(32, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x, _ = self.lstm(x)
        # Max pooling across the time sequence
        x, _ = torch.max(x, dim=1) 
        x = self.dropout(self.relu(self.fc1(x)))
        return self.fc2(x)


def main():
    if not os.path.exists(NUMPY_OUTPUT_PATH):
        print(f"Error: Dataset folder '{NUMPY_OUTPUT_PATH}' not found. Please ensure process_dataset.py has finished successfully.")
        return

    actions = np.array(sorted([d for d in os.listdir(NUMPY_OUTPUT_PATH) if os.path.isdir(os.path.join(NUMPY_OUTPUT_PATH, d))]))
    if len(actions) == 0:
        print("No class folders found in dataset.")
        return
        
    print("Training on classes:", actions)

    sequences, labels = [], []
    for label, action in enumerate(actions):
        action_path = os.path.join(NUMPY_OUTPUT_PATH, action)
        files = [f for f in os.listdir(action_path) if f.endswith('.npy')]
        
        for file in files:
            res_path = os.path.join(action_path, file)
            res = np.load(res_path) # Shape: (30, 546)
            sequences.append(res)
            labels.append(label)
            
    X = np.array(sequences)
    y = np.array(labels)
    
    if len(X) == 0:
        print("Error: No data found.")
        return
        
    print(f"Loaded dataset: {len(X)} samples. Feature dimension: {X.shape[2]}")

    # Stratified Split (80/20)
    try:
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    except ImportError:
        print("Error: scikit-learn not installed. Please run 'pip install scikit-learn'.")
        return

    train_dataset = ISLDataset(X_train, y_train, augment=True)
    val_dataset = ISLDataset(X_val, y_val, augment=False)

    # Class Weight Balancer
    class_counts = Counter(y_train)
    weights = [1.0 / class_counts[i] for i in y_train]
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    print(f"Train samples: {len(X_train)} | Val samples: {len(X_val)}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Training on device:", device)

    model = ASLModel(input_size=X.shape[2], hidden_size=64, num_classes=len(actions)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    EPOCHS = 150
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    history = {'train_acc': [], 'val_acc': [], 'loss': []}

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        train_correct = 0
        train_total = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += batch_y.size(0)
            train_correct += (predicted == batch_y).sum().item()
            
        scheduler.step()
        train_acc = 100 * train_correct / train_total if train_total > 0 else 0
            
        # Eval every epoch
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                _, predicted = torch.max(outputs.data, 1)
                val_total += batch_y.size(0)
                val_correct += (predicted == batch_y).sum().item()
        
        val_acc = 100 * val_correct / val_total if val_total > 0 else 0
        
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['loss'].append(total_loss/len(train_loader))
        
        if (epoch+1) % 10 == 0 or epoch == 0:
            print(f'Epoch [{epoch+1}/{EPOCHS}], Loss: {history["loss"][-1]:.4f}, Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%')

    # Plotting
    try:
        plt.figure(figsize=(10, 5))
        plt.plot(history['train_acc'], label='Train Accuracy')
        plt.plot(history['val_acc'], label='Val Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.savefig('accuracy_curve.png')
        print("Saved accuracy plot to accuracy_curve.png")
    except Exception as e:
        print(f"Skipping plot generation: {e}")

    torch.save({
        'model_state_dict': model.state_dict(),
        'classes': actions.tolist()
    }, 'action.pt')
    print("Training Complete! Saved to action.pt")

if __name__ == "__main__":
    main()
