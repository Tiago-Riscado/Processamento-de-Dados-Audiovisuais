import numpy as np
import pandas as pd
import os
from PIL import Image
import torch
from torchvision import transforms, models
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

BASE_DIR = r"D:\Universidade\Lic_IACD\2023-2024\2Semestre\1Semestre\Programacao\Linguagens\Python\projeto\Processamento-de-Dados-Audiovisuais\data\dataset_split\test"
OUTPUT_CSV = r'D:\Universidade\Lic_IACD\2023-2024\2Semestre\1Semestre\Programacao\Linguagens\Python\projeto\Processamento-de-Dados-Audiovisuais\Metricas\scores_modelo.csv'
MODEL_PATH = r"C:\Users\tiago\Desktop\best_model_MobilNet.pth"

IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = sorted([folder for folder in os.listdir(BASE_DIR)
                      if os.path.isdir(os.path.join(BASE_DIR, folder))])

CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}
IDX_TO_CLASS = {v: k for k, v in CLASS_TO_IDX.items()}

print("\nClasses encontradas:")
for c, idx in CLASS_TO_IDX.items():
    print(f"{idx} → {c}")

NUM_CLASSES = len(CLASS_TO_IDX)

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# -------------------------
# Carregar MobileNetV2
# -------------------------
model = models.mobilenet_v2(weights=None)  # sem pesos pré-treinados
model.classifier = nn.Sequential(
    nn.Dropout(0.4),
    nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
)

model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# -------------------------
# Função para classificar uma imagem
# -------------------------
def classify_image(img_path):
    img = Image.open(img_path).convert('RGB')
    img = transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        outputs = model(img)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
        pred_class = np.argmax(probs)
    return pred_class, probs

# -------------------------
# Função principal
# -------------------------
def run_test():
    all_paths = []
    for root, _, files in os.walk(BASE_DIR):
        for filename in files:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                all_paths.append(os.path.join(root, filename))

    if not all_paths:
        print(f"Nenhuma imagem encontrada em '{BASE_DIR}'.")
        return

    print(f"\nTotal de {len(all_paths)} imagens encontradas.")

    y_true = []
    y_pred = []
    results_list = []

    for path in all_paths:
        pred_class, probs = classify_image(path)

        class_name = os.path.basename(os.path.dirname(path))
        true_class = CLASS_TO_IDX[class_name]

        y_true.append(true_class)
        y_pred.append(pred_class)

        row = {
            'Imagem': os.path.basename(path),
            'Classe_Verdadeira': true_class,
            'Previsao': pred_class
        }

        for i, score in enumerate(probs):
            row[f'Classe_{i}'] = float(score)

        results_list.append(row)

    df_results = pd.DataFrame(results_list)
    df_results.to_csv(OUTPUT_CSV, index=False)
    print(f"\nResultados exportados para: {OUTPUT_CSV}")

    acc = accuracy_score(y_true, y_pred)
    print(f"\nAccuracy Global: {acc*100:.2f}%")

    print("\nRelatório por classe:\n")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))

    print("\nMatriz de Confusão:\n")
    print(confusion_matrix(y_true, y_pred))

if __name__ == '__main__':
    os.makedirs(BASE_DIR, exist_ok=True)
    run_test()
