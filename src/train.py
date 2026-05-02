import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from src.config import NUM_EPOCHS, LEARNING_RATE, PATIENCE


def train_model(model, train_loader, val_loader, train_dataset, val_dataset,
                model_save_path: str, device: torch.device) -> dict:
    """
    Loop de treino com early stopping.
    Devolve o histórico de loss/accuracy para plotting.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE,
    )

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_loss = float("inf")
    counter       = 0
    best_wts      = None

    for epoch in range(1, NUM_EPOCHS + 1):

        # --- Treino ---
        model.train()
        running_loss, running_corrects = 0.0, 0
        for inputs, labels in tqdm(train_loader, desc=f"Época {epoch}/{NUM_EPOCHS} [Train]", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss    = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss     += loss.item() * inputs.size(0)
            running_corrects += torch.sum(torch.argmax(outputs, 1) == labels).item()

        epoch_loss = running_loss     / len(train_dataset)
        epoch_acc  = running_corrects / len(train_dataset)

        # --- Validação ---
        model.eval()
        val_loss, val_corrects = 0.0, 0
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f"Época {epoch}/{NUM_EPOCHS} [Val]", leave=False):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs    = model(inputs)
                val_loss  += criterion(outputs, labels).item() * inputs.size(0)
                val_corrects += torch.sum(torch.argmax(outputs, 1) == labels).item()

        epoch_val_loss = val_loss     / len(val_dataset)
        epoch_val_acc  = val_corrects / len(val_dataset)

        history["train_loss"].append(epoch_loss)
        history["val_loss"].append(epoch_val_loss)
        history["train_acc"].append(epoch_acc)
        history["val_acc"].append(epoch_val_acc)

        print(f"Época {epoch}/{NUM_EPOCHS} | "
              f"Train Loss: {epoch_loss:.4f} | Val Loss: {epoch_val_loss:.4f} | "
              f"Train Acc: {epoch_acc:.4f} | Val Acc: {epoch_val_acc:.4f}")

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            counter       = 0
            best_wts      = model.state_dict()
            torch.save(best_wts, model_save_path)
            print("  → Modelo guardado")
        else:
            counter += 1
            if counter >= PATIENCE:
                print("Early stopping ativado")
                break

    if best_wts is not None:
        model.load_state_dict(best_wts)

    return history
