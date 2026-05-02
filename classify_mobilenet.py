"""
Entry point — Classificação com MobileNetV2 no conjunto de teste.

Uso:
    python classify_mobilenet.py
"""

import os
import torch
import torch.nn as nn
from torchvision import models

from src.config   import MODEL_MOBILENET, OUTPUT_CSV_MOBILENET, OUTPUT_DIR
from src.classify import get_class_names, run_classification

DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TEST_DIR = os.path.join(OUTPUT_DIR, "test")

class_names = get_class_names(TEST_DIR)
num_classes = len(class_names)

print("\nClasses encontradas:")
for idx, name in enumerate(class_names):
    print(f"  {idx} → {name}")

# --- Carregar modelo ---
model = models.mobilenet_v2(weights=None)
model.classifier = nn.Sequential(
    nn.Dropout(0.4),
    nn.Linear(model.classifier[1].in_features, num_classes),
)
model.load_state_dict(torch.load(MODEL_MOBILENET, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# --- Classificar ---
run_classification(model, TEST_DIR, OUTPUT_CSV_MOBILENET, class_names, DEVICE)
