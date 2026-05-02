import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from src.dataset import pad_to_square, get_val_transforms
from src.config  import IMG_SIZE, RESULTS_DIR
import os


def evaluate_model(model, test_loader, test_dataset, classes: list, device: torch.device):
    """Avalia no conjunto de teste e imprime métricas."""
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            preds = torch.argmax(model(inputs), 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print(f"\nTest Accuracy: {accuracy_score(all_labels, all_preds):.4f}")
    print(classification_report(all_labels, all_preds, target_names=classes, digits=4))

    _plot_confusion_matrix(all_labels, all_preds, classes)
    return all_labels, all_preds


def _plot_confusion_matrix(y_true, y_pred, classes, fname="confusion_matrix.png"):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=classes, yticklabels=classes, cmap="Blues")
    plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, fname), dpi=150)
    plt.show()


def plot_training_curves(history: dict, fname="training_curves.png"):
    r = range(1, len(history["train_loss"]) + 1)
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(r, history["train_loss"], label="Train Loss")
    plt.plot(r, history["val_loss"],   label="Val Loss")
    plt.xlabel("Épocas"); plt.ylabel("Loss"); plt.legend(); plt.grid(True)
    plt.title("Curva de Loss")

    plt.subplot(1, 2, 2)
    plt.plot(r, history["train_acc"], label="Train Accuracy")
    plt.plot(r, history["val_acc"],   label="Val Accuracy")
    plt.xlabel("Épocas"); plt.ylabel("Accuracy"); plt.legend(); plt.grid(True)
    plt.title("Curva de Accuracy")

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, fname), dpi=150)
    plt.show()


def plot_gradcam(model, target_layer, test_dataset, classes: list,
                 device: torch.device, num_examples: int = 12, fname="gradcam.png"):
    """Visualiza Grad-CAM em imagens aleatórias do test set."""
    val_transforms = get_val_transforms()
    cam      = GradCAM(model=model, target_layers=[target_layer])
    selected = random.sample(test_dataset.samples, min(num_examples, len(test_dataset.samples)))

    cols = 4
    rows = (len(selected) + cols - 1) // cols
    plt.figure(figsize=(cols * 4, rows * 3))

    for i, (path, label) in enumerate(selected):
        img_pil    = Image.open(path).convert("RGB")
        img_sq     = pad_to_square(img_pil).resize((IMG_SIZE, IMG_SIZE))
        img_np     = np.array(img_sq).astype(np.float32) / 255.0
        input_tensor = val_transforms(img_sq).unsqueeze(0).to(device)

        model.eval()
        with torch.no_grad():
            pred_class = torch.argmax(model(input_tensor)).item()

        grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0]
        cam_image     = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)

        ax = plt.subplot(rows, cols, i + 1)
        ax.imshow(cam_image); ax.axis("off")
        ax.set_title(f"True: {classes[label]}\nPred: {classes[pred_class]}")

    plt.suptitle("Grad-CAM: imagens aleatórias do test set")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, fname), dpi=150)
    plt.show()
