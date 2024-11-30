import optuna
import pymysql
import urllib.parse
import time
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from datetime import datetime
import config

# URL de connexion à la base de données
storage_url = f"mysql+pymysql://{config.USER}:{urllib.parse.quote(config.PASSWORD)}@{config.ENDPOINT}/{config.DATABASE_NAME}"

# Configuration pour WandB
wandb.init(project="benchmark-optuna-mysql", name="torch_optimization", config={"epochs": 10, "batch_size": 256})
wandb_config = wandb.config

# Appareil utilisé
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Appareil utilisé : {device}")


# Modèle simple pour le test
class SimpleModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.fc(x)


# Données synthétiques
def generate_data(batch_size=256, input_size=1000):
    x = torch.randn(batch_size, input_size)
    y = torch.randint(0, 10, (batch_size,))
    return x, y


# Fonction d'entraînement
def train(model, device, optimizer, criterion, epochs=10, batch_size=256):
    model.to(device)
    for epoch in range(epochs):
        epoch_loss = 0.0
        for _ in range(10):  # Simulation de 10 batches par epoch
            optimizer.zero_grad()
            x, y = generate_data(batch_size, input_size=1000)
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        # Log dans WandB
        avg_loss = epoch_loss / 10
        wandb.log({"epoch": epoch + 1, "loss": avg_loss})
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")


# Fonction d'objectif pour Optuna
def objective(trial):
    # Hyperparamètres suggérés
    hidden_size = trial.suggest_int("hidden_size", 128, 1024)
    lr = trial.suggest_loguniform("lr", 1e-4, 1e-2)
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])

    # Configuration du modèle et des optimisateurs
    model = SimpleModel(input_size=1000, hidden_size=hidden_size, output_size=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Entraînement
    start_time = time.time()
    train(model, device, optimizer, criterion, epochs=wandb_config.epochs, batch_size=batch_size)
    end_time = time.time()

    # Calcul du temps total
    total_time = end_time - start_time
    wandb.log({"total_time": total_time, "hidden_size": hidden_size, "lr": lr, "batch_size": batch_size})
    print(f"Temps total : {total_time:.2f} secondes")

    # Retourner une métrique à maximiser (inverse du temps ici)
    return 1 / total_time


if __name__ == "__main__":
    # Création d'une étude avec stockage dans MySQL
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(storage_url)
    study = optuna.create_study(
        direction="maximize",
        storage=storage_url,
        study_name=f"VastAI - {current_time}",
        sampler=optuna.samplers.TPESampler(seed=1)
    )

    # Optimisation
    study.optimize(objective, n_trials=10)  # 10 essais pour le test

    # Résultats finaux
    print("Meilleurs hyperparamètres : ", study.best_params)
    print("Valeur maximale obtenue : ", study.best_value)
    wandb.finish()
