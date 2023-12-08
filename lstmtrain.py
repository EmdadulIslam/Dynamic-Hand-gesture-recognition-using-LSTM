import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import os

# Define the LSTM network architecture
class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, num_layers):
        super(LSTMClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device) 
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

class CustomDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.Tensor(features.values)
        self.labels, self.label_mapping = pd.factorize(labels)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

data = pd.read_csv('Resfeatures_df_HGD2_aug.csv')

features = data.iloc[:, :-2]
labels = data['label']

train_indices = data[data['split'] == 'train'].index
test_indices = data[data['split'] == 'test'].index

train_features = features.loc[train_indices]
train_labels = labels.loc[train_indices]
test_features = features.loc[test_indices]
test_labels = labels.loc[test_indices]

train_dataset = CustomDataset(train_features, train_labels)
test_dataset = CustomDataset(test_features, test_labels)

input_size = len(features.columns)
hidden_size = 64
num_classes = len(np.unique(labels))
num_layers = 1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = LSTMClassifier(input_size, hidden_size, num_classes, num_layers).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

batch_size = 32
num_epochs = 100

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

train_loss_history = []
val_loss_history = []
accuracy_history = []

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for batch_features, batch_labels in train_loader:
        batch_features = batch_features.unsqueeze(1).to(device)  # Add an extra dimension
        batch_labels = batch_labels.to(device)
        outputs = model(batch_features)
        loss = criterion(outputs, batch_labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
    
    train_loss /= len(train_loader)
    train_loss_history.append(train_loss)

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch_features, batch_labels in test_loader:
            batch_features = batch_features.unsqueeze(1).to(device)  # Add an extra dimension
            batch_labels = batch_labels.to(device)
            outputs = model(batch_features)
            val_loss += criterion(outputs, batch_labels).item()

    val_loss /= len(test_loader)
    val_loss_history.append(val_loss)

    total_correct = 0
    total_samples = 0
    predictions = []
    real_labels = []
    with torch.no_grad():
        for batch_features, batch_labels in test_loader:
            batch_features = batch_features.unsqueeze(1).to(device)  # Add an extra dimension
            batch_labels = batch_labels.to(device)
            outputs = model(batch_features)
            _, predicted = torch.max(outputs.data, 1)
            predictions.extend(predicted.cpu().numpy())
            real_labels.extend(batch_labels.cpu().numpy())
            total_samples += batch_labels.size(0)
            total_correct += (predicted == batch_labels).sum().item()

    accuracy = total_correct / total_samples
    accuracy_history.append(accuracy)

    print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {train_loss}, Validation Loss: {val_loss}, Accuracy: {accuracy}")


torch.save(model.state_dict(), 'LSTMmodel.pth')

print("Label Mapping (Original Labels to Encoded Values):")
for idx, label in enumerate(train_dataset.label_mapping):
    print(f"{label}: {idx}")

save_dir = "E:\\LSTMmodel"

os.makedirs(save_dir, exist_ok=True)

plt.figure()
plt.plot(range(1, num_epochs + 1), train_loss_history, label='Train Loss')
plt.plot(range(1, num_epochs + 1), val_loss_history, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss vs. Epoch')
plt.legend()
plt.savefig(os.path.join(save_dir, "loss_vs_epoch.png"))
plt.close()

plt.figure()
plt.plot(range(1, num_epochs + 1), accuracy_history, label='Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy vs. Epoch')
plt.legend()
plt.savefig(os.path.join(save_dir, "accuracy_vs_epoch.png"))
plt.close()

plt.figure()
cm = confusion_matrix(real_labels, predictions)
sns.heatmap(cm, annot=True)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.savefig(os.path.join(save_dir, "confusion_matrix.png"))
plt.close()

overall_accuracy = np.mean(accuracy_history)
print(f'Overall Accuracy: {overall_accuracy}')
