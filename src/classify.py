import os
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from src.dataset import get_val_transforms
from src.config  import IMG_SIZE


def get_class_names(test_dir: str) -> list:
    return sorted([
        d for d in os.listdir(test_dir)
        if os.path.isdir(os.path.join(test_dir, d))
    ])


def classify_image(model, img_path: str, transform, device: torch.device):
    img = Image.open(img_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1).cpu().numpy()[0]
    return int(np.argmax(probs)), probs


def run_classification(model, test_dir: str, output_csv: str,
                       class_names: list, device: torch.device):
    """
    Percorre test_dir, classifica todas as imagens e exporta CSV + métricas.
    """
    transform = get_val_transforms()

    all_paths = [
        os.path.join(root, f)
        for root, _, files in os.walk(test_dir)
        for f in files
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))
    ]

    if not all_paths:
        print(f"Nenhuma imagem encontrada em '{test_dir}'.")
        return

    print(f"\nTotal de {len(all_paths)} imagens encontradas.")

    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    y_true, y_pred, rows = [], [], []

    for path in all_paths:
        pred_class, probs = classify_image(model, path, transform, device)
        true_name  = os.path.basename(os.path.dirname(path))
        true_class = class_to_idx[true_name]

        y_true.append(true_class)
        y_pred.append(pred_class)

        row = {"Imagem": os.path.basename(path),
               "Classe_Verdadeira": true_class,
               "Previsao": pred_class}
        for i, score in enumerate(probs):
            row[f"Classe_{i}"] = float(score)
        rows.append(row)

    pd.DataFrame(rows).to_csv(output_csv, index=False)
    print(f"Resultados exportados para: {output_csv}")

    print(f"\nAccuracy Global: {accuracy_score(y_true, y_pred)*100:.2f}%")
    print(classification_report(y_true, y_pred, target_names=class_names))
    print("Matriz de Confusão:")
    print(confusion_matrix(y_true, y_pred))
