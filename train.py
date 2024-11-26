import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Générer des données factices
def generate_data(num_samples=1000):
    X = torch.randn(num_samples, 10)  # 10 features
    y = (torch.sum(X, dim=1) > 0).float()  # Classe 1 si somme des features > 0, sinon 0
    return X, y

# Modèle simple
class SimpleModel(nn.Module):
    def __init__(self, input_size):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(input_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.fc(x))

# Configuration
input_size = 10
batch_size = 32
num_epochs = 5
learning_rate = 0.01

# Charger les données
X, y = generate_data()
dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialiser le modèle, la loss et l'optimiseur
model = SimpleModel(input_size)
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# Entraînement
print("Début de l'entraînement...")
for epoch in range(num_epochs):
    epoch_loss = 0.0
    for batch_X, batch_y in dataloader:
        optimizer.zero_grad()
        outputs = model(batch_X).squeeze()
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print(f"Époque {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}")

# Sauvegarde du modèle
torch.save(model.state_dict(), "model.pth")
print("Modèle sauvegardé dans model.pth.")
