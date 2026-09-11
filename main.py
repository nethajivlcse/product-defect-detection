"""
Automated Product Defect Detection - Binary Image Classification
Computer Vision Practical Assessment

Pipeline: Dataset Prep -> Transfer Learning (ResNet18) -> Training ->
          Evaluation (metrics + confusion matrix) -> Error Analysis

Expected folder structure for --data_dir:
    dataset/
        Normal/
            img1.jpg
            ...
        Defective/
            img1.jpg
            ...

Outputs are written into:
    model/trained_model.pth
    results/training_curves.png
    results/confusion_matrix.png
    results/misclassified_examples.png
    results/metrics.txt

Usage:
    python main.py --data_dir ./dataset --epochs 8
"""

import os
import argparse
import copy
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix, ConfusionMatrixDisplay)

# ----------------------------------------------------------------------------
# Reproducibility + device selection
# ----------------------------------------------------------------------------
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()   # Apple Silicon GPU
    else "cpu"
)


# ----------------------------------------------------------------------------
# TASK 1: Dataset Preparation
# ----------------------------------------------------------------------------
def get_dataloaders(data_dir, img_size=224, batch_size=32, val_split=0.2):
    """
    Loads images with torchvision.datasets.ImageFolder (expects data_dir to
    contain one subfolder per class, e.g. data_dir/Normal, data_dir/Defective).

    - Resizes every image to 224x224 and converts to a tensor.
    - Normalizes using ImageNet mean/std because we use an ImageNet-pretrained
      backbone -- the pretrained weights expect inputs in that distribution.
    - Augmentation (flip / rotation / crop / brightness-contrast) is applied
      ONLY to the training split, never to validation, per the assessment.
    - The 80/20 split is stratified so both classes are proportionally
      represented in both the train and validation sets.
    """
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomRotation(degrees=15),
        transforms.RandomCrop(img_size, padding=8, padding_mode="reflect"),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std),
    ])

    # Load the folder twice with different transforms, then take matching
    # Subset indices. This lets train/val use different augmentation while
    # guaranteeing they point at the exact same underlying images/labels.
    base_dataset = datasets.ImageFolder(data_dir)  # only used for labels/classes
    targets = [label for _, label in base_dataset.samples]

    train_idx, val_idx = train_test_split(
        np.arange(len(targets)),
        test_size=val_split,
        stratify=targets,        # keeps class balance equal in both splits
        random_state=SEED,
    )

    train_dataset_full = datasets.ImageFolder(data_dir, transform=train_transform)
    val_dataset_full = datasets.ImageFolder(data_dir, transform=val_transform)

    train_dataset = Subset(train_dataset_full, train_idx)
    val_dataset = Subset(val_dataset_full, val_idx)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                               shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                             shuffle=False, num_workers=0)

    class_names = base_dataset.classes  # e.g. ['Defective', 'Normal']
    print(f"Classes: {class_names}")
    print(f"Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")

    return train_loader, val_loader, class_names, val_idx, base_dataset


# ----------------------------------------------------------------------------
# TASK 2: Transfer Learning Model
# ----------------------------------------------------------------------------
def build_model(num_classes=2, freeze_backbone=True):
    """
    ResNet18 pretrained on ImageNet, used as a fixed feature extractor.
    - All convolutional (backbone) layers are frozen: requires_grad=False.
    - Only the final fully-connected layer is replaced and left trainable.

    Why this design: with a limited training budget and a small dataset,
    freezing the backbone means we only optimize a small number of
    parameters, which converges in a handful of epochs and resists
    overfitting, while still benefiting from ImageNet's learned visual
    features (edges, textures, shapes).
    """
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False  # freeze pretrained conv layers

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)  # new head, trainable by default

    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    frozen_count = sum(1 for p in model.parameters() if not p.requires_grad)
    print(f"Trainable layers: {trainable}")
    print(f"Frozen parameter tensors: {frozen_count}")

    return model.to(DEVICE)


# ----------------------------------------------------------------------------
# TASK 3: Training
# ----------------------------------------------------------------------------
def train_model(model, train_loader, val_loader, epochs=8, lr=1e-3):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr
    )

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_acc = 0.0
    best_weights = copy.deepcopy(model.state_dict())

    for epoch in range(epochs):
        # ---- training phase ----
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / total
        train_acc = correct / total

        # ---- validation phase (no gradient updates - val data never trains) ----
        model.eval()
        val_running_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_running_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_loss = val_running_loss / val_total
        val_acc = val_correct / val_total

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            best_weights = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_weights)
    print(f"Best validation accuracy: {best_acc:.4f}")
    return model, history


def plot_history(history, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(history["train_loss"], label="Train Loss")
    axes[0].plot(history["val_loss"], label="Val Loss")
    axes[0].set_title("Loss vs Epoch")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(history["train_acc"], label="Train Acc")
    axes[1].plot(history["val_acc"], label="Val Acc")
    axes[1].set_title("Accuracy vs Epoch")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved training curves to {out_path}")
    plt.close()


# ----------------------------------------------------------------------------
# PART 2: Evaluation
# ----------------------------------------------------------------------------
def evaluate_model(model, val_loader, class_names, out_path, metrics_txt_path):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(DEVICE)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)

    report_lines = [
        "--- Validation Metrics ---",
        f"Accuracy : {acc:.4f}",
        f"Precision: {precision:.4f}",
        f"Recall   : {recall:.4f}",
        f"F1-Score : {f1:.4f}",
        "Confusion Matrix (rows=actual, cols=predicted):",
        f"Classes: {class_names}",
        str(cm),
    ]
    report_text = "\n".join(report_lines)
    print("\n" + report_text)

    # Save metrics to a text file in results/ so they're visible alongside plots
    with open(metrics_txt_path, "w") as f:
        f.write(report_text + "\n")
    print(f"Saved metrics to {metrics_txt_path}")

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix")
    plt.savefig(out_path, dpi=150)
    print(f"Saved confusion matrix to {out_path}")
    plt.close()

    metrics = {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}
    return metrics, all_preds, all_labels


# ----------------------------------------------------------------------------
# PART 2: Error Analysis
# ----------------------------------------------------------------------------
def error_analysis(model, base_dataset, val_idx, class_names, out_path, num_examples=3):
    """
    Re-runs inference on raw validation images (kept unnormalized for display)
    and shows the first `num_examples` misclassified samples with their
    actual vs predicted labels and model confidence.
    """
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]
    display_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    norm_transform = transforms.Normalize(imagenet_mean, imagenet_std)

    model.eval()
    misclassified = []

    with torch.no_grad():
        for idx in val_idx:
            path, true_label = base_dataset.samples[idx]
            image = base_dataset.loader(path)
            display_tensor = display_transform(image)
            input_tensor = norm_transform(display_tensor).unsqueeze(0).to(DEVICE)

            output = model(input_tensor)
            confidence = torch.softmax(output, dim=1).max().item()
            _, pred = torch.max(output, 1)
            pred = pred.item()

            if pred != true_label:
                misclassified.append((display_tensor, true_label, pred, confidence, path))
            if len(misclassified) >= num_examples:
                break

    if not misclassified:
        print("No misclassified validation images found.")
        return []

    fig, axes = plt.subplots(1, len(misclassified), figsize=(5 * len(misclassified), 5))
    if len(misclassified) == 1:
        axes = [axes]

    for ax, (img_tensor, true_label, pred, conf, path) in zip(axes, misclassified):
        img_np = img_tensor.permute(1, 2, 0).numpy()
        ax.imshow(img_np)
        ax.set_title(f"Actual: {class_names[true_label]}\n"
                      f"Predicted: {class_names[pred]}\n"
                      f"Confidence: {conf:.2f}")
        ax.axis("off")
        print(f"\nMisclassified: {os.path.basename(path)}")
        print(f"  Actual: {class_names[true_label]} | Predicted: {class_names[pred]} "
              f"| Confidence: {conf:.2f}")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"\nSaved misclassified examples to {out_path}")
    plt.close()

    return misclassified


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./dataset",
                         help="Path to dataset folder with one subfolder per class")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--model_dir", type=str, default="./model",
                         help="Folder where the trained model (.pth) is saved")
    parser.add_argument("--results_dir", type=str, default="./results",
                         help="Folder where plots and metrics are saved")
    args = parser.parse_args()

    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    print(f"Using device: {DEVICE}")

    train_loader, val_loader, class_names, val_idx, base_dataset = get_dataloaders(
        args.data_dir, batch_size=args.batch_size
    )

    model = build_model(num_classes=len(class_names), freeze_backbone=True)

    model, history = train_model(model, train_loader, val_loader,
                                  epochs=args.epochs, lr=args.lr)

    plot_history(history, out_path=os.path.join(args.results_dir, "training_curves.png"))

    evaluate_model(
        model, val_loader, class_names,
        out_path=os.path.join(args.results_dir, "confusion_matrix.png"),
        metrics_txt_path=os.path.join(args.results_dir, "metrics.txt"),
    )

    error_analysis(
        model, base_dataset, val_idx, class_names,
        out_path=os.path.join(args.results_dir, "misclassified_examples.png"),
        num_examples=3,
    )

    model_path = os.path.join(args.model_dir, "trained_model.pth")
    torch.save(model.state_dict(), model_path)
    print(f"\nSaved trained model to {model_path}")


if __name__ == "__main__":
    main()
