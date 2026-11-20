"""
Evaluation & Benchmark Script for Crop Disease Detection Model
Computes Confusion Matrix, Precision, Recall, F1-Score, and generates visual plots.
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.cnn_model import get_model


def evaluate_model(data_dir: str = None, model_path: str = None, device: str = None):
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = data_dir or os.path.join(project_dir, "data", "dataset")
    model_path = model_path or os.path.join(project_dir, "models", "saved_models", "crop_disease_model.pth")
    saved_models_dir = os.path.join(project_dir, "models", "saved_models")

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating model on {device}...")

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    dataset = datasets.ImageFolder(data_dir, transform=val_transform)
    data_loader = DataLoader(dataset, batch_size=32, shuffle=False)
    class_names = dataset.classes

    # Load model checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    model_type = checkpoint.get("model_type", "cnn") if isinstance(checkpoint, dict) else "cnn"
    state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint

    model = get_model(model_type, num_classes=len(class_names), pretrained=False)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    acc = accuracy_score(all_targets, all_preds) * 100.0
    print(f"Overall Dataset Accuracy: {acc:.2f}%")

    clean_labels = [c.replace("___", " - ").replace("_", " ") for c in class_names]

    # Classification Report
    clf_report = classification_report(all_targets, all_preds, target_names=clean_labels, output_dict=True)
    report_text = classification_report(all_targets, all_preds, target_names=clean_labels)
    print("\n--- Classification Report ---")
    print(report_text)

    # Confusion Matrix
    cm = confusion_matrix(all_targets, all_preds)
    plt.figure(figsize=(14, 11))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=clean_labels, yticklabels=clean_labels)
    plt.title("Crop Disease CNN - Confusion Matrix", fontsize=15, pad=15)
    plt.xlabel("Predicted Class", fontsize=12)
    plt.ylabel("Ground Truth Class", fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()

    cm_path = os.path.join(saved_models_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=200)
    plt.close()
    print(f"Confusion matrix saved to: {cm_path}")

    # Save evaluation report JSON
    eval_report = {
        "overall_accuracy": round(acc, 2),
        "total_samples": len(all_targets),
        "detailed_metrics": clf_report
    }
    eval_json_path = os.path.join(saved_models_dir, "evaluation_report.json")
    with open(eval_json_path, "w", encoding="utf-8") as f:
        json.dump(eval_report, f, indent=2)

    return eval_report


if __name__ == "__main__":
    evaluate_model()
