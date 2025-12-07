import os
import random
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns
import shutil
from tqdm import tqdm
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

# Config
data_dir = r"D:\Universidade\Lic_IACD\2023-2024\2Semestre\1Semestre\Programacao\Linguagens\Python\projeto\Processamento-de-Dados-Audiovisuais\data\dataset_waste_container"
balanced_dir = r"D:\Universidade\Lic_IACD\2023-2024\2Semestre\1Semestre\Programacao\Linguagens\Python\projeto\Processamento-de-Dados-Audiovisuais\data\dataset_augmented"
output_dir = r"D:\Universidade\Lic_IACD\2023-2024\2Semestre\1Semestre\Programacao\Linguagens\Python\projeto\Processamento-de-Dados-Audiovisuais\data\dataset_split"
img_size = 224
target_images_per_class = 1000
batch_size = 32
num_epochs = 30
learning_rate = 0.0005
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
val_images = 200
test_images = 200

# Quantidade inicial de imagens por classe

print("Contagem inicial de imagens por classe no diretório original:")
all_classes = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]

for c in sorted(all_classes):
    class_path = os.path.join(data_dir, c)
    imgs = [f for f in os.listdir(class_path) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    print(f"{c}: {len(imgs)}")

# Remover classe container_ash

class_to_remove = "container_ash"

for path in [os.path.join(data_dir, class_to_remove), os.path.join(balanced_dir, class_to_remove)]:
    if os.path.exists(path):
        shutil.rmtree(path)

for split in ["train", "val", "test"]:
    split_path = os.path.join(output_dir, split, class_to_remove)
    if os.path.exists(split_path):
        shutil.rmtree(split_path)

# Padding

def pad_to_square(img, fill_color=(0,0,0)):
    w, h = img.size
    if w == h:
        return img
    max_side = max(w, h)
    new_img = Image.new("RGB", (max_side, max_side), fill_color)
    new_img.paste(img, ((max_side - w) // 2, (max_side - h) // 2))
    return new_img

# Traforms e data augmentation
augment_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(40),
    transforms.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8),
    transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0))
])
 
os.makedirs(balanced_dir, exist_ok=True)
classes = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]

for c in classes:
    os.makedirs(os.path.join(balanced_dir, c), exist_ok=True)
    images = [f for f in os.listdir(os.path.join(data_dir, c)) if f.lower().endswith((".jpg",".jpeg",".png"))]

    if len(images) >= target_images_per_class:
        selected_images = random.sample(images, target_images_per_class)
    else:
        selected_images = images

    for img_file in selected_images:
        img = Image.open(os.path.join(data_dir, c, img_file)).convert("RGB")
        img = pad_to_square(img)
        img = img.resize((img_size, img_size))
        img.save(os.path.join(balanced_dir, c, img_file))

    n_to_generate = target_images_per_class - len(selected_images)
    for i in range(n_to_generate):
        img_file = random.choice(images)
        img = Image.open(os.path.join(data_dir, c, img_file)).convert("RGB")
        img = pad_to_square(img)
        img_aug = augment_transforms(img)
        img_aug.save(os.path.join(balanced_dir, c, f"aug_{i}_{img_file}.jpg"))

# Verificar contagem após Data Augmentation

print("\nContagem de imagens por classe após Data Augmentation:")
for c in classes:
    class_path = os.path.join(balanced_dir, c)
    imgs = [f for f in os.listdir(class_path) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    print(f"{c}: {len(imgs)}")

# Divisão dos conjuntos
os.makedirs(output_dir, exist_ok=True)
for split in ["train", "val", "test"]:
    for c in classes:
        os.makedirs(os.path.join(output_dir, split, c), exist_ok=True)

for c in classes:
    images = [f for f in os.listdir(os.path.join(balanced_dir, c)) if f.lower().endswith((".jpg",".jpeg",".png"))]
    random.shuffle(images)
    train_imgs = images[:len(images) - val_images - test_images]
    val_imgs = images[len(images) - val_images - test_images: len(images) - test_images]
    test_imgs = images[len(images) - test_images:]

    for img_file in train_imgs:
        shutil.copy(os.path.join(balanced_dir, c, img_file), os.path.join(output_dir, "train", c, img_file))
    for img_file in val_imgs:
        shutil.copy(os.path.join(balanced_dir, c, img_file), os.path.join(output_dir, "val", c, img_file))
    for img_file in test_imgs:
        shutil.copy(os.path.join(balanced_dir, c, img_file), os.path.join(output_dir, "test", c, img_file))

# Normalização da ImageNet
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std  = [0.229, 0.224, 0.225]

train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
])
val_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(img_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
])

# DataLoaders
train_dataset = datasets.ImageFolder(os.path.join(output_dir, "train"), transform=train_transforms)
val_dataset = datasets.ImageFolder(os.path.join(output_dir, "val"), transform=val_transforms)
test_dataset = datasets.ImageFolder(os.path.join(output_dir, "test"), transform=val_transforms)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

classes = train_dataset.classes
num_classes = len(classes)

# Modelo ResNet18 com Fine-Tuning
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(model.fc.in_features, num_classes)  #

for param in model.parameters():
    param.requires_grad = False
for param in model.layer4.parameters():
    param.requires_grad = True
for param in model.fc.parameters():
    param.requires_grad = True

model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)

# Treino com Early Stopping
best_val_loss = float('inf')
patience = 5
counter = 0
best_model_wts = None
train_losses, val_losses = [], []
train_accs, val_accs = [], []

for epoch in range(num_epochs):
    model.train()
    running_loss, running_corrects = 0.0, 0
    for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", leave=False):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(torch.argmax(outputs, 1) == labels.data)
    epoch_loss = running_loss / len(train_dataset)
    epoch_acc = running_corrects.double() / len(train_dataset)
    
    # Validação
    model.eval()
    val_running_loss, val_corrects = 0.0, 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_running_loss += loss.item() * inputs.size(0)
            val_corrects += torch.sum(torch.argmax(outputs, 1) == labels.data)
    val_loss = val_running_loss / len(val_dataset)
    val_acc = val_corrects.double() / len(val_dataset)

    train_losses.append(epoch_loss)
    val_losses.append(val_loss)
    train_accs.append(epoch_acc.item())
    val_accs.append(val_acc.item())

    print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {epoch_loss:.4f} | Val Loss: {val_loss:.4f} | Train Acc: {epoch_acc:.4f} | Val Acc: {val_acc:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        counter = 0
        best_model_wts = model.state_dict()
        torch.save(best_model_wts, "best_resnet18_model.pth")
    else:
        counter += 1
        if counter >= patience:
            print("Early stopping")
            break

if best_model_wts is not None:
    model.load_state_dict(best_model_wts)

# Avaliação no conjunto de teste
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        preds = torch.argmax(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

accuracy = accuracy_score(all_labels, all_preds)
print("Test Accuracy:", accuracy)
print(classification_report(all_labels, all_preds, target_names=classes, digits=4))

# Plots das curvas de treino e validação
def plot_training_curves(train_losses, val_losses, train_acc, val_acc):
    plt.figure(figsize=(10,5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Épocas')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Curva de Loss')
    plt.show()

    plt.figure(figsize=(10,5))
    plt.plot(train_acc, label='Train Accuracy')
    plt.plot(val_acc, label='Val Accuracy')
    plt.xlabel('Épocas')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Curva de Acurácia')
    plt.show()

plot_training_curves(train_losses, val_losses, train_accs, val_accs)

# Matriz de Confusão
cmatrix = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(8,6))
sns.heatmap(cmatrix, annot=True, fmt="d", xticklabels=classes, yticklabels=classes, cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# Grad-CAM em 12 imagens aleatórias do test set
target_layer = model.layer4[-1].conv2  # última convolução da layer4
cam = GradCAM(model=model, target_layers=[target_layer])

num_examples = 12
samples = test_dataset.samples
selected = random.sample(samples, min(num_examples, len(samples)))

cols = 4
rows = (len(selected) + cols - 1) // cols
plt.figure(figsize=(cols * 4, rows * 3))

for i, (path, label) in enumerate(selected):
    img_pil = Image.open(path).convert('RGB')
    img_sq = pad_to_square(img_pil)
    img_resized = img_sq.resize((img_size, img_size))
    
    img_np = np.array(img_resized).astype(np.float32) / 255.0
    input_tensor = val_transforms(img_resized).unsqueeze(0).to(device)
    
    input_tensor.requires_grad = True
    model.eval()
    with torch.no_grad():
        pred_class = torch.argmax(model(input_tensor)).item()

    grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0]
    cam_image = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)
    
    ax = plt.subplot(rows, cols, i+1)
    ax.imshow(cam_image)
    ax.axis('off')
    ax.set_title(f"True: {classes[label]}\nPred: {classes[pred_class]}")

plt.suptitle('Grad-CAM: 12 imagens aleatórias do test set')
plt.tight_layout()
plt.show()
