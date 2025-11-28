import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from config import Config

def train_model(model, train_loader, val_loader):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    best_val_loss = float("inf")
    device = Config.DEVICE

    print("\nStep 3: Starting CNN Training...\n")

    for epoch in range(Config.NUM_EPOCHS):
        model.train()
        train_loss = 0
        correct = 0
        total = 0

        # training
        for X, y in tqdm(train_loader, desc=f"Epoch {epoch+1} Train", leave=False):
            X, y = X.to(device), y.to(device)

            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()

        train_loss /= len(train_loader)
        train_acc = 100 * correct / total

        # validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for X, y in tqdm(val_loader, desc=f"Epoch {epoch+1} Val", leave=False):
                X, y = X.to(device), y.to(device)
                outputs = model(X)

                loss = criterion(outputs, y)
                val_loss += loss.item()

                _, predicted = outputs.max(1)
                val_total += y.size(0)
                val_correct += predicted.eq(y).sum().item()

        val_loss /= len(val_loader)
        val_acc = 100 * val_correct / val_total

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            print(f"Validation improved {best_val_loss:.4f} → {val_loss:.4f}. Saving...")
            best_val_loss = val_loss
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)

        # logging
        print(
            f"Epoch {epoch+1}/{Config.NUM_EPOCHS} | "
            f"Train Loss {train_loss:.4f} | Train Acc {train_acc:.2f}% | "
            f"Val Loss {val_loss:.4f} | Val Acc {val_acc:.2f}% | "
            f"LR: {optimizer.param_groups[0]['lr']:.6f}"
        )
