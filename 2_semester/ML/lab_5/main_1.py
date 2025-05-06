import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

x = np.linspace(0, 4 * np.pi, 100)
y = np.sin(x)

scaler = MinMaxScaler()
y_scaled = scaler.fit_transform(y.reshape(-1, 1)).flatten()

window_size = 10
sequences = np.array([y_scaled[i:i+window_size] for i in range(len(y_scaled) - window_size)])

X_tensor = torch.tensor(sequences, dtype=torch.float32)
dataset = TensorDataset(X_tensor, X_tensor)
loader = DataLoader(dataset, batch_size=8, shuffle=True)

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(window_size, 6),
            nn.ReLU(),
            nn.Linear(6, 3)
        )
        self.decoder = nn.Sequential(
            nn.Linear(3, 6),
            nn.ReLU(),
            nn.Linear(6, window_size)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

model = Autoencoder()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

epochs = 100
for epoch in range(epochs):
    for batch_x, _ in loader:
        output = model(batch_x)
        loss = criterion(output, batch_x)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.6f}")

model.eval()
with torch.no_grad():
    reconstructed = model(X_tensor).numpy()

mse = mean_squared_error(sequences, reconstructed)
print(f"Среднеквадратичная ошибка (MSE): {mse:.6f}")

for i in range(3):
    plt.plot(sequences[i], label='Original (смещен)', linestyle='--', alpha=0.8)
    plt.plot(reconstructed[i], label='Reconstructed')
    plt.title(f"Окно {i}")
    plt.legend()
    plt.show()
