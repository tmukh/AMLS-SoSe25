import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score

def evaluate(model, loader, DEVICE, ECG_1D_CNN_Enhanced=None):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for signals, lengths, labels in loader:
            signals, lengths, labels = signals.to(DEVICE), lengths.to(DEVICE), labels.to(DEVICE)
            if ECG_1D_CNN_Enhanced and isinstance(model, ECG_1D_CNN_Enhanced):
                outputs = model(signals)
            else:
                outputs = model(signals, lengths)
            preds = torch.argmax(outputs, dim=1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
    f1 = f1_score(y_true, y_pred, average='macro')
    return f1

def train_model(model, train_loader, val_loader, class_weights, DEVICE, epochs=50, patience=15):
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    best_val_f1 = 0
    patience_counter = 0
    history = {'train_loss': [], 'val_f1': []}

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for signals, lengths, labels in train_loader:
            signals, lengths, labels = signals.to(DEVICE), lengths.to(DEVICE), labels.to(DEVICE)
            outputs = model(signals, lengths)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        val_f1 = evaluate(model, val_loader, DEVICE)
        history['train_loss'].append(avg_loss)
        history['val_f1'].append(val_f1)

        print(f"Epoch {epoch+1} - Loss: {avg_loss:.4f}, Val F1: {val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping.")
                break

    return history
