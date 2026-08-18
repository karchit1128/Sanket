import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

# 1. Configuration
DATA_PATH = os.path.join(os.path.dirname(__file__), 'MP_Data')
actions = np.array(['hello', 'thanks', 'iloveyou'])
no_sequences = 30
sequence_length = 30
input_size = 1662
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
    for sequence in np.array(os.listdir(os.path.join(DATA_PATH, action))).astype(int):
        window = []
        for frame_num in range(sequence_length):
            res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])

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
        # x is (batch_size, sequence_length, input_size)
        out, _ = self.lstm(x)
        # Get the output from the last time step
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
