"""
Model Training Pipeline for Crop Disease Detection Using CNN
Handles data augmentation, validation splits, learning rate scheduling,
early stopping, metrics calculation, and serialization of trained weights.
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

# Ensure project root is in sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.cnn_model import get_model


def get_data_loaders(data_dir: str, batch_size: int = 32, val_split: float = 0.2):
    """
    Creates augmented train and validation data loaders.
    """
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    full_dataset = datasets.ImageFolder(data_dir, transform=train_transform)
    class_names = full_dataset.classes

    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size

    train_data, val_data = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    # Apply validation transforms to validation set
    val_data.dataset.transform = val_transform

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader, class_names


def train_model(data_dir: str,
                model_type: str = "cnn",
                epochs: int = 15,
                batch_size: int = 32,
                learning_rate: float = 0.001,
                device: str = None):
    
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} | Model Architecture: {model_type.upper()}")

    train_loader, val_loader, class_names = get_data_loaders(data_dir, batch_size=batch_size)
    num_classes = len(class_names)
    print(f"Total Classes: {num_classes}")

    model = get_model(model_type, num_classes=num_classes, pretrained=True)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    history = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": []
    }

    best_val_acc = 0.0
    models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "saved_models")
    os.makedirs(models_dir, exist_ok=True)
    
    save_path = os.path.join(models_dir, "crop_disease_model.pth")
    class_names_path = os.path.join(models_dir, "class_names.json")
    metrics_path = os.path.join(models_dir, "metrics.json")

    with open(class_names_path, "w", encoding="utf-8") as f:
        json.dump(class_names, f, indent=2)

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        # Training Phase
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct_train += torch.sum(preds == labels.data).item()
            total_train += labels.size(0)

        epoch_train_loss = running_loss / total_train
        epoch_train_acc = (correct_train / total_train) * 100.0

        # Validation Phase
        model.eval()
        val_running_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_running_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                correct_val += torch.sum(preds == labels.data).item()
                total_val += labels.size(0)

        epoch_val_loss = val_running_loss / max(1, total_val)
        epoch_val_acc = (correct_val / max(1, total_val)) * 100.0

        scheduler.step(epoch_val_loss)

        history["train_loss"].append(round(epoch_train_loss, 4))
        history["train_acc"].append(round(epoch_train_acc, 2))
        history["val_loss"].append(round(epoch_val_loss, 4))
        history["val_acc"].append(round(epoch_val_acc, 2))

        print(f"Epoch [{epoch:02d}/{epochs:02d}] "
              f"Train Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc:.2f}% | "
              f"Val Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc:.2f}%")

        if epoch_val_acc >= best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save({
                "model_type": model_type,
                "state_dict": model.state_dict(),
                "num_classes": num_classes,
                "epoch": epoch,
                "best_val_acc": best_val_acc
            }, save_path)

    elapsed = time.time() - start_time
    print(f"\nTraining completed in {elapsed:.1f}s. Best Val Accuracy: {best_val_acc:.2f}%")
    print(f"Saved optimal model weights to: {save_path}")

    # Plot & Save Learning Curves
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history["train_loss"], label="Train Loss", color="royalblue")
    plt.plot(history["val_loss"], label="Val Loss", color="crimson")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Cross Entropy Loss Curve")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)

    plt.subplot(1, 2, 2)
    plt.plot(history["train_acc"], label="Train Acc", color="forestgreen")
    plt.plot(history["val_acc"], label="Val Acc", color="darkorange")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Model Accuracy Curve")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    curves_path = os.path.join(models_dir, "training_curves.png")
    plt.savefig(curves_path, dpi=200)
    plt.close()

    # Save metrics JSON
    metrics = {
        "model_type": model_type,
        "epochs": epochs,
        "best_val_acc": round(best_val_acc, 2),
        "final_train_acc": history["train_acc"][-1],
        "final_train_loss": history["train_loss"][-1],
        "final_val_loss": history["val_loss"][-1],
        "history": history,
        "training_time_seconds": round(elapsed, 2)
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return model, metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Crop Disease CNN")
    parser.add_argument("--epochs", type=int, default=12, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--model", type=str, default="cnn", choices=["cnn", "mobilenet"], help="Model architecture")
    args = parser.parse_args()

    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_path = os.path.join(project_dir, "data", "dataset")

    train_model(
        data_dir=dataset_path,
        model_type=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr
    )
