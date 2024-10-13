'''
Use various sequential and non-sequential networks in order to predict your alcohol level from your cell phone's
accelerometer. Use the BarCrawl dataset. Experiment with various options such as activation functions, number of layers,
number of hidden nodes per layer, optimization algorithms, loss functions, sequence length, batch size and type of
target variable.

For each option plot the loss function over several epochs. Submit your notebook and present your best results.
'''

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import bc_data as bc
import matplotlib.pyplot as plt
from sklearn import preprocessing
import numpy as np


class NormalizedBarCrawlDataset(bc.BarCrawlDataset):
    def __init__(self, seq_size):
        super().__init__(seq_size)
        self.scaler = preprocessing.StandardScaler()

        # collect all accelerometer data to fit the scaler
        all_data = []
        for i in range(len(self)):
            x, _ = super().__getitem__(i)
            all_data.append(x.numpy())
        all_data = np.vstack(all_data)

        # fit the scaler
        self.scaler.fit(all_data)

    def __getitem__(self, idx):
        x, y = super().__getitem__(idx)
        # transform the data
        x_normalized = self.scaler.transform(x.numpy())
        return torch.FloatTensor(x_normalized), y


class LSTMNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.2):
        super(LSTMNet, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


# hyperparameters
seq_size = 100
batch_size = 64
input_size = 3
hidden_size = 96
num_layers = 2
num_epochs = 50
learning_rate = 0.001

# load and split the dataset
dataset = NormalizedBarCrawlDataset(seq_size)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# initialize network and optimizer
net = LSTMNet(input_size, hidden_size, num_layers)
criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=learning_rate)

# training and evaluation
train_losses = []
test_losses = []

for epoch in range(num_epochs):
    # training phase
    net.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        labels = labels.float()

        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(torch.squeeze(outputs), labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    train_loss = running_loss / len(train_loader)
    train_losses.append(train_loss)

    # evaluation phase
    net.eval()
    test_loss = 0.0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs
            labels = labels.float()

            outputs = net(inputs)
            loss = criterion(torch.squeeze(outputs), labels)
            test_loss += loss.item()

    test_loss = test_loss / len(test_loader)
    test_losses.append(test_loss)

    print(f'Epoch [{epoch + 1}/{num_epochs}], '
          f'Train Loss: {train_loss:.4f}, '
          f'Test Loss: {test_loss:.4f}')

# plot the results
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Training Loss')
plt.plot(test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and test loss over epochs')
plt.legend()
plt.grid(True)
plt.savefig('loss_plot.png')
plt.show()

