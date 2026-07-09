import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd

from model import Autoencoder

uc_matrix = np.load('./data/train_uc.npy')
uc_tensor = torch.Tensor(uc_matrix)
input_dim_uc = uc_tensor.shape[1]
hidden_dim_uc = 128
autoencoder_uc = Autoencoder(input_dim_uc, hidden_dim_uc)

criterion = nn.MSELoss()
optimizer_uc = optim.Adam(autoencoder_uc.parameters(), lr=0.01)
batch_size = 512
uc_loader = DataLoader(TensorDataset(uc_tensor, uc_tensor), batch_size=batch_size, shuffle=True)
num_epochs = 100

def train_autoencoder(model, data_loader, optimizer, criterion, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        for data in data_loader:
            inputs, _ = data
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

train_autoencoder(autoencoder_uc, uc_loader, optimizer_uc, criterion, num_epochs)

autoencoder_uc.eval()
user_course_features = autoencoder_uc.encoder(uc_tensor).detach().numpy()


np.save('./data/user_course_features.npy', user_course_features)
