import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler
from collections import Counter

# Set this to the folder containing the extracted .npy files
NUMPY_OUTPUT_PATH = 'dataset' 

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
        seq_ids = set([f.split('_')[0] for f in files])
        
        for sequence_id in seq_ids:
            window = []
            for frame_num in range(30): # 30 FRAMES (3 SECONDS at 10 FPS)
                res_path = os.path.join(action_path, f"{sequence_id}_{frame_num}.npy")
                if os.path.exists(res_path):
                    res = np.load(res_path)
                    window.append(res)
                else:
                    if len(window) > 0:
                        window.append(window[-1])
                    else:
                        window.append(np.zeros(1692))
            sequences.append(window)
            labels.append(label)
            
    X = np.array(sequences)
    y = np.array(labels)

    class ISLDataset(Dataset):
        def __init__(self, X, y, augment=False):
            self.X = torch.FloatTensor(X)
            self.y = torch.LongTensor(y)
            self.augment = augment
            
        def __len__(self):
            return len(self.X)
            
        def __getitem__(self, idx):
            x_data = self.X[idx]
            # Gaussian Noise Augmentation for robustness
            if self.augment:
                noise = torch.randn_like(x_data) * 0.005
                x_data = x_data + noise
            return x_data, self.y[idx]

    # Train/Val Split (80/20)
    dataset_size = len(X)
    train_size = int(0.8 * dataset_size)
    val_size = dataset_size - train_size
    
    if dataset_size == 0:
        print("Error: No data found.")
        return

    full_dataset = list(zip(X, y))
    train_data, val_data = random_split(full_dataset, [train_size, val_size])

    X_train, y_train = zip(*train_data)
    if val_size > 0:
        X_val, y_val = zip(*val_data)
    else:
        # If very few samples, use train for val to prevent crashing
        print("Warning: Dataset too small for validation split. Using train set for validation.")
        X_val, y_val = X_train, y_train
        val_size = train_size

    train_dataset = ISLDataset(X_train, y_train, augment=True)
    val_dataset = ISLDataset(X_val, y_val, augment=False)

    # Class Weight Balancer
    class_counts = Counter(y_train)
    weights = [1.0 / class_counts[i] for i in y_train]
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    print(f"Train samples: {train_size} | Val samples: {val_size}")

    class ASLModel(nn.Module):
        def __init__(self, input_size, hidden_size, num_layers, num_classes):
            super().__init__()
            self.lstm1 = nn.LSTM(input_size=258, hidden_size=64, batch_first=True)
            self.lstm2 = nn.LSTM(input_size=64, hidden_size=128, batch_first=True)
            self.lstm3 = nn.LSTM(input_size=128, hidden_size=64, batch_first=True)
            self.dropout = nn.Dropout(0.3) # Dropout to prevent overfitting
            self.fc1 = nn.Linear(64, 64)
            self.fc2 = nn.Linear(64, 32)
            self.fc3 = nn.Linear(32, num_classes)
            self.relu = nn.ReLU()

        def forward(self, x):
            x, _ = self.lstm1(x)
            x, _ = self.lstm2(x)
            x, _ = self.lstm3(x)
            x = x[:, -1, :]
            x = self.dropout(self.relu(self.fc1(x)))
            x = self.relu(self.fc2(x))
            return self.fc3(x)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Training on device:", device)

    model = ASLModel(258, 64, 3, len(actions)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Cosine Annealing Learning Rate Scheduler
    EPOCHS = 150
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        scheduler.step()
            
        if (epoch+1) % 10 == 0:
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
            print(f'Epoch [{epoch+1}/{EPOCHS}], Loss: {total_loss/len(train_loader):.4f}, Val Accuracy: {val_acc:.2f}%')

    torch.save({
        'model_state_dict': model.state_dict(),
        'classes': actions.tolist()
    }, 'action.pt')
    print("Training Complete! Saved to action.pt")

if __name__ == "__main__":
    main()
