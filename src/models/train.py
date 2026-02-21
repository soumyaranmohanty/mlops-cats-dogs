import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
import torch.optim as optim
from src.data.preprocess import get_data_loaders
from src.models.model import SimpleCNN
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np

def train():
    mlflow.set_experiment("cats_vs_dogs")
    train_losses = []
    val_losses = []

    with mlflow.start_run():

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        train_loader, val_loader, test_loader = get_data_loaders("data/raw")

        model = SimpleCNN().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        epochs = 5

        # 🔹 Log parameters
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("learning_rate", 0.001)
        mlflow.log_param("batch_size", 32)

        for epoch in range(epochs):
            model.train()
            running_loss = 0.0

            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            avg_loss = running_loss / len(train_loader)

            # 🔹 Validation
            model.eval()
            val_loss = 0.0

            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()

            val_loss /= len(val_loader)

            print(f"Epoch {epoch+1}, Train Loss: {avg_loss:.4f}, Val Loss: {val_loss:.4f}")

            train_losses.append(avg_loss)
            val_losses.append(val_loss)

            # 🔹 Log metrics
            mlflow.log_metric("train_loss", avg_loss, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)

        
        plt.figure()
        plt.plot(train_losses, label="Train Loss")
        plt.plot(val_losses, label="Validation Loss")
        plt.legend()
        plt.title("Loss Curve")

        plt.savefig("loss_curve.png")
        mlflow.log_artifact("loss_curve.png")

        model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                outputs = model(images)
                preds = torch.argmax(outputs, dim=1).cpu().numpy()

                all_preds.extend(preds)
                all_labels.extend(labels.numpy())

        cm = confusion_matrix(all_labels, all_preds)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()

        plt.savefig("confusion_matrix.png")
        mlflow.log_artifact("confusion_matrix.png")
        # 🔹 Save model
        torch.save(model.state_dict(), "model.pt")

        # 🔹 Log model
        mlflow.pytorch.log_model(model, "model")

        print("Model saved and logged!")

if __name__ == "__main__":
    train()