"""
Entry point — Treino e avaliação com MobileNetV2.

Uso:
    python main_mobilenet.py
"""

import torch
import torch.nn as nn
from torchvision import models

from src.config   import MODEL_MOBILENET
from src.dataset  import remove_class, balance_and_augment, split_dataset, get_dataloaders
from src.train    import train_model
from src.evaluate import evaluate_model, plot_training_curves, plot_gradcam

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)

# --- Preparação dos dados ---
remove_class()
balance_and_augment()
split_dataset()

train_loader, val_loader, test_loader, classes = get_dataloaders()
num_classes = len(classes)

# --- Modelo MobileNetV2 com fine-tuning ---
model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
model.classifier = nn.Sequential(
    nn.Dropout(0.4),
    nn.Linear(model.classifier[1].in_features, num_classes),
)

for param in model.features.parameters():
    param.requires_grad = False
for block in model.features[-5:]:
    for param in block.parameters():
        param.requires_grad = True
for param in model.classifier.parameters():
    param.requires_grad = True

model = model.to(DEVICE)

# --- Treino ---
history = train_model(
    model, train_loader, val_loader,
    train_loader.dataset, val_loader.dataset,
    MODEL_MOBILENET, DEVICE,
)

# --- Avaliação ---
plot_training_curves(history, fname="mobilenet_training_curves.png")
evaluate_model(model, test_loader, test_loader.dataset, classes, DEVICE)
plot_gradcam(
    model,
    target_layer=model.features[-1],
    test_dataset=test_loader.dataset,
    classes=classes,
    device=DEVICE,
    fname="mobilenet_gradcam.png",
)
