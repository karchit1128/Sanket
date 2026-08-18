import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

# 1. Configuration
DATA_PATH = os.path.join(os.path.dirname(__file__), 'MP_Data')

if not os.path.exists(DATA_PATH):
    print("Error: MP_Data folder not found. Please run collect_data.py or process_dataset.py first.")
    exit()

# Dynamically find all actions based on folder names
actions = np.array([d for d in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, d))])
if len(actions) == 0:
    print("Error: No data found in MP_Data.")
    exit()

print(f"Found {len(actions)} actions to train on: {actions}")

# Save actions.txt for vision.py to use
with open(os.path.join(os.path.dirname(__file__), 'actions.txt'), 'w') as f:
    for action in actions:
        f.write(f"{action}\n")

sequence_length = 30
input_size = 1692
hidden_size = 64
num_layers = 3
num_classes = len(actions)
num_epochs = 150
batch_size = 16
learning_rate = 0.001

label_map = {label:num for num, label in enumerate(actions)}

# 2. Load Data
sequences, labels = [], []
for action in actions:
    for sequence_folder in os.listdir(os.path.join(DATA_PATH, action)):
        seq_path = os.path.join(DATA_PATH, action, sequence_folder)
        if not os.path.isdir(seq_path): continue
        
        window = []
        # Check if the sequence has exactly sequence_length frames
        if len(os.listdir(seq_path)) < sequence_length:
            continue
            
        for frame_num in range(sequence_length):
            res = np.load(os.path.join(seq_path, "{}.npy".format(frame_num)))
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])

if len(sequences) == 0:
    print("Error: No valid 30-frame sequences found.")
    exit()

X = np.array(sequences)
y = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

# Convert to PyTorch tensors
X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.long)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.long)

train_dataset = TensorDataset(X_train_t, y_train_t)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# 3. Define the PyTorch Model
class ASLModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(ASLModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, num_classes)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out

model = ASLModel(input_size, hidden_size, num_layers, num_classes)

# 4. Training
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

print("Starting training...")
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')

# 5. Evaluate
model.eval()
with torch.no_grad():
    outputs = model(X_test_t)
    _, predicted = torch.max(outputs.data, 1)
    correct = (predicted == y_test_t).sum().item()
    accuracy = correct / len(y_test_t)
    print(f'Test Accuracy: {accuracy * 100:.2f}%')

# 6. Save Model
torch.save(model.state_dict(), 'action.pt')
print("Model saved successfully as 'action.pt'!")
