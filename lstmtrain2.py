import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import numpy as np
import glob
import os

# Step 1: Data Preparation
feature_paths = glob.glob("E:\\LSTMmodel\\HGD2full\\*\\*\\*\\*Res3d_feature.npy")
"E:\LSTMmodel\HGD2full"
# Load encoded feature vectors and corresponding labels
data = [np.load(path, allow_pickle=True) for path in feature_paths]
features = np.concatenate([item for item in data], axis=0)

# Extract labels from the paths
labels = [os.path.normpath(path).split(os.sep)[-3] for path in feature_paths]

# Convert the labels to integer
le = LabelEncoder()
labels = le.fit_transform(labels)

# Print encoded labels
for class_name, class_label in zip(le.classes_, range(len(le.classes_))):
    print(f'Label: {class_name}, Encoded: {class_label}')

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(features, labels, test_size=0.2, random_state=42)

# Set the input size based on the feature vector shape
input_size = features.shape[1]

# Step 2: Define the LSTM Model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # Add a batch dimension if it's missing
        if len(x.shape) == 2:
            x = x.unsqueeze(0)

        # Initialize hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)

        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out

# Step 3: Data Loading for Training
class CustomDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        # Adding an extra dimension for sequence length
        return self.features[index][np.newaxis, :], self.labels[index]

batch_size = 32  # Set the batch size for training
train_dataset = CustomDataset(X_train, y_train)
val_dataset = CustomDataset(X_val, y_val)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# Step 4: Training the LSTM Model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
hidden_size = 128  # Choose the number of hidden units in the LSTM
num_layers = 2  # Choose the number of LSTM layers
num_classes = len(np.unique(labels))  # Set the number of gesture classes
learning_rate = 0.001  # Set the learning rate
num_epochs = 100  # Set the number of training epochs

model = LSTMModel(input_size, hidden_size, num_layers, num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

train_losses = []
val_losses = []
val_accuracies = []

for epoch in range(num_epochs):
    model.train()
    for features, labels in train_loader:
        features, labels = features.to(device), labels.to(device)

        outputs = model(features.float())
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_losses.append(loss.item())

    # Step 5: Validation
    model.eval()
    with torch.no_grad():
        val_loss = 0
        correct = 0
        total = 0
        for features, labels in val_loader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features.float())
            val_loss += criterion(outputs, labels).item()
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

        val_losses.append(val_loss / len(val_loader))
        val_accuracy = correct / total
        val_accuracies.append(val_accuracy)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Validation Loss: {val_loss / len(val_loader):.4f}, Validation Accuracy: {val_accuracy:.4f}')

# Step 6: Save the Model
torch.save(model.state_dict(), 'LSTMmodel_vc.pth')

# Step 7: Loss Curves
plt.figure(figsize=(7, 5))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training vs Validation Loss')
plt.grid(True)
plt.savefig('loss_curve.png')  # Save the plot
plt.show()

# Step 8: Accuracy Curve
plt.figure(figsize=(7, 5))
plt.plot(val_accuracies, label='Validation Accuracy')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Validation Accuracy')
plt.grid(True)
plt.savefig('accuracy_curve.png')  # Save the plot
plt.show()

print(f'Overall accuracy: {val_accuracies[-1]:.4f}')

