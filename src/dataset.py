import os
import random
import shutil
from PIL import Image
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

from src.config import (
    DATA_DIR, BALANCED_DIR, OUTPUT_DIR,
    IMG_SIZE, TARGET_IMAGES_PER_CLASS,
    VAL_IMAGES, TEST_IMAGES, CLASS_TO_REMOVE,
    BATCH_SIZE, IMAGENET_MEAN, IMAGENET_STD,
)


# ------------------------------------------------------------------ #
# Utilitários de imagem
# ------------------------------------------------------------------ #

def pad_to_square(img: Image.Image, fill_color=(0, 0, 0)) -> Image.Image:
    """Adiciona padding para tornar a imagem quadrada."""
    w, h = img.size
    if w == h:
        return img
    max_side = max(w, h)
    new_img = Image.new("RGB", (max_side, max_side), fill_color)
    new_img.paste(img, ((max_side - w) // 2, (max_side - h) // 2))
    return new_img


# ------------------------------------------------------------------ #
# Pipeline de preparação de dados
# ------------------------------------------------------------------ #

def remove_class(cls: str = CLASS_TO_REMOVE):
    """Remove a classe indesejada dos directorios de dados."""
    for path in [os.path.join(DATA_DIR, cls), os.path.join(BALANCED_DIR, cls)]:
        if os.path.exists(path):
            shutil.rmtree(path)
    for split in ["train", "val", "test"]:
        p = os.path.join(OUTPUT_DIR, split, cls)
        if os.path.exists(p):
            shutil.rmtree(p)


def balance_and_augment():
    """Equilibra as classes com data augmentation e guarda em BALANCED_DIR."""
    augment = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(40),
        transforms.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8),
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
    ])

    os.makedirs(BALANCED_DIR, exist_ok=True)
    classes = [
        d for d in os.listdir(DATA_DIR)
        if os.path.isdir(os.path.join(DATA_DIR, d)) and d != CLASS_TO_REMOVE
    ]

    for c in classes:
        os.makedirs(os.path.join(BALANCED_DIR, c), exist_ok=True)
        images = [
            f for f in os.listdir(os.path.join(DATA_DIR, c))
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        selected = random.sample(images, min(len(images), TARGET_IMAGES_PER_CLASS))
        for img_file in selected:
            img = Image.open(os.path.join(DATA_DIR, c, img_file)).convert("RGB")
            img = pad_to_square(img).resize((IMG_SIZE, IMG_SIZE))
            img.save(os.path.join(BALANCED_DIR, c, img_file))

        n_gen = TARGET_IMAGES_PER_CLASS - len(selected)
        for i in range(n_gen):
            img_file = random.choice(images)
            img = Image.open(os.path.join(DATA_DIR, c, img_file)).convert("RGB")
            img = pad_to_square(img)
            img_aug = augment(img)
            img_aug.save(os.path.join(BALANCED_DIR, c, f"aug_{i}_{img_file}.jpg"))


def split_dataset():
    """Divide BALANCED_DIR em train / val / test."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    classes = [
        d for d in os.listdir(BALANCED_DIR)
        if os.path.isdir(os.path.join(BALANCED_DIR, d))
    ]

    for split in ["train", "val", "test"]:
        for c in classes:
            os.makedirs(os.path.join(OUTPUT_DIR, split, c), exist_ok=True)

    for c in classes:
        images = [
            f for f in os.listdir(os.path.join(BALANCED_DIR, c))
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        random.shuffle(images)

        n_train = TARGET_IMAGES_PER_CLASS - VAL_IMAGES - TEST_IMAGES
        train_imgs = images[:n_train]
        val_imgs   = images[n_train: n_train + VAL_IMAGES]
        test_imgs  = images[n_train + VAL_IMAGES:]

        for split_name, split_imgs in [("train", train_imgs), ("val", val_imgs), ("test", test_imgs)]:
            for img_file in split_imgs:
                shutil.copy(
                    os.path.join(BALANCED_DIR, c, img_file),
                    os.path.join(OUTPUT_DIR, split_name, c, img_file),
                )


# ------------------------------------------------------------------ #
# Transforms
# ------------------------------------------------------------------ #

def get_train_transforms():
    return transforms.Compose([
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_val_transforms():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# ------------------------------------------------------------------ #
# DataLoaders
# ------------------------------------------------------------------ #

def get_dataloaders():
    train_ds = datasets.ImageFolder(os.path.join(OUTPUT_DIR, "train"), transform=get_train_transforms())
    val_ds   = datasets.ImageFolder(os.path.join(OUTPUT_DIR, "val"),   transform=get_val_transforms())
    test_ds  = datasets.ImageFolder(os.path.join(OUTPUT_DIR, "test"),  transform=get_val_transforms())

    return (
        DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True),
        DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False),
        DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False),
        train_ds.classes,
    )
