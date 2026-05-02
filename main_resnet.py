"""
Entry point — Treino e avaliação com ResNet18.

Uso:
    python main_resnet.py
"""

import torch
import torch.nn as nn
from torchvision import models

from src.config  import MODEL_RESNET, RESULTS_DIR
from src.dataset import remove_class, balance_and_augment, split_dataset, get_dataloaders
from src.train   import train_model
from src.evaluate import evaluate_model, plot_training_curves, plot_gradcam

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)

# --- Preparação dos dados ---
remove_class()
balance_and_augment()
split_dataset()

train_loader, val_loader, test_loader, classes = get_dataloaders()
num_classes = len(classes)

# --- Modelo ResNet18 com fine-tuning ---
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(model.fc.in_features, num_classes)

for param in model.parameters():
    param.requires_grad = False
for param in model.layer4.parameters():
    param.requires_grad = True
for param in model.fc.parameters():
    param.requires_grad = True

model = model.to(DEVICE)

# --- Treino ---
history = train_model(
    model, train_loader, val_loader,
    train_loader.dataset, val_loader.dataset,
    MODEL_RESNET, DEVICE,
)

# --- Avaliação ---
plot_training_curves(history, fname="resnet_training_curves.png")
evaluate_model(model, test_loader, test_loader.dataset, classes, DEVICE)
plot_gradcam(
    model,
    target_layer=model.layer4[-1].conv2,
    test_dataset=test_loader.dataset,
    classes=classes,
    device=DEVICE,
    fname="resnet_gradcam.png",
)
