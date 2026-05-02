# Waste Container Classifier

Classificação de contentores de resíduos com fine-tuning de ResNet18 e MobileNetV2, com visualização Grad-CAM.

## Estrutura

```
waste-classifier/
│
├── main_resnet.py          # Treino + avaliação ResNet18
├── main_mobilenet.py       # Treino + avaliação MobileNetV2
├── classify_resnet.py      # Inferência com ResNet18
├── classify_mobilenet.py   # Inferência com MobileNetV2
│
├── src/
│   ├── config.py           # Caminhos e hiperparâmetros (lê .env)
│   ├── dataset.py          # Augmentation, balancing, split, DataLoaders
│   ├── train.py            # Loop de treino com early stopping
│   ├── evaluate.py         # Métricas, confusion matrix, Grad-CAM
│   └── classify.py         # Lógica de inferência partilhada
│
├── models/                 # Pesos .pth — não versionados
├── results/                # Gráficos e CSVs — não versionados
├── data/                   # Dataset — não versionado
│
├── requirements.txt
├── .env.example
└── .gitignore
```

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env        # ajustar caminhos se necessário
```

## Utilização

```bash
# Treinar
python main_resnet.py
python main_mobilenet.py

# Classificar (requer pesos em models/)
python classify_resnet.py
python classify_mobilenet.py
```

## Modelos

| Modelo       | Fine-tuning         |
|-------------|---------------------|
| ResNet18    | layer4 + fc         |
| MobileNetV2 | últimas 5 features + classifier |

## Pipeline de dados

1. Remove classe `container_ash`
2. Data augmentation até 1000 imagens por classe
3. Split: 600 train / 200 val / 200 test
