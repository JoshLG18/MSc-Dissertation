# ----------------- Script for training the DL models ----------------------
# --------------------------------------------------------------------------
# import all the libraries
# Import all model modules
# Import training utilities, data loaders, metrics
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm.auto import tqdm
import optuna
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, precision_score, recall_score, f1_score, confusion_matrix
import json
import os
import pandas as pd

# --------------------------------------------------------------------------
def train_one_epoch(model, dataloader, optimizer, criterion, device, grad_clip=1.0):
    model.train()
    epoch_loss = 0
    for xb, yb in dataloader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out.squeeze(), yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        epoch_loss += loss.item() * xb.size(0)
    return epoch_loss / len(dataloader.dataset)

def validate_one_epoch(model, dataloader, criterion, device):
    model.eval()
    epoch_loss = 0
    preds, targets = [], []
    for xb, yb in dataloader:
        xb, yb = xb.to(device), yb.to(device)
        out = model(xb)
        loss = criterion(out.squeeze(), yb)
        epoch_loss += loss.item() * xb.size(0)
        preds.extend(out.squeeze().detach().cpu().numpy())
        targets.extend(yb.detach().cpu().numpy())
    return epoch_loss / len(dataloader.dataset), np.array(preds), np.array(targets)

def train_full_model(
    model, train_loader, val_loader, optimizer, criterion, scheduler, device,
    epochs=100, grad_clip=1.0, early_stopping_patience=10, save_path=None
):
    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0
    history = {"train_loss": [], "val_loss": []}
    for epoch in tqdm(range(epochs), desc="Training (epochs)", dynamic_ncols=True):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, grad_clip)
        val_loss = None
        if val_loader is not None:
            val_loss, _, _ = validate_one_epoch(model, val_loader, criterion, device)
            scheduler.step(val_loss)
        else:
            scheduler.step(train_loss)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.6f}" + (f" | Val Loss: {val_loss:.6f}" if val_loss is not None else ""))
        if val_loss is not None and val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict()
            patience_counter = 0
            if save_path:
                torch.save(best_state, save_path)
        elif val_loss is not None:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print("Early stopping triggered.")
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history

def get_dataloaders(X_train, y_train, X_val, y_val, batch_size, device, seq_len=1):
    def create_sequences(X, y, seq_len):
        Xs, ys = [], []
        for i in range(len(X) - seq_len + 1):
            Xs.append(X[i:i+seq_len])
            ys.append(y[i+seq_len-1])
        return np.array(Xs), np.array(ys)
    X_train_seq, y_train_seq = create_sequences(X_train, y_train, seq_len)
    X_val_seq, y_val_seq = create_sequences(X_val, y_val, seq_len)
    X_train_tensor = torch.tensor(X_train_seq, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train_seq, dtype=torch.float32).to(device)
    X_val_tensor = torch.tensor(X_val_seq, dtype=torch.float32).to(device)
    y_val_tensor = torch.tensor(y_val_seq, dtype=torch.float32).to(device)
    train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(TensorDataset(X_val_tensor, y_val_tensor), batch_size=batch_size, shuffle=False)
    return train_loader, val_loader

def evaluate_model(model, test_loader, device):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            preds.extend(out.squeeze().detach().cpu().numpy())
            targets.extend(yb.detach().cpu().numpy())
    preds = np.array(preds)
    targets = np.array(targets)
    # Directional metrics
    d_pred, d_true = np.sign(preds), np.sign(targets)
    dir_acc = float(np.mean(d_pred == d_true))
    precision = float(precision_score(d_true, d_pred, labels=[-1, 1], average='macro', zero_division=0))
    recall = float(recall_score(d_true, d_pred, labels=[-1, 1], average='macro', zero_division=0))
    f1 = float(f1_score(d_true, d_pred, labels=[-1, 1], average='macro', zero_division=0))
    conf_matrix = confusion_matrix(d_true, d_pred, labels=[-1, 1]).tolist()
    metrics = {
        "mse": mean_squared_error(targets, preds),
        "mae": mean_absolute_error(targets, preds),
        "r2": r2_score(targets, preds),
        "dir_acc": dir_acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": conf_matrix
    }
    return {
        "actual_returns": targets,
        "predicted_returns": preds,
        "test_metrics": metrics
    }

def save_test_predictions(y_true, y_pred, out_dir, filename):
    df = pd.DataFrame({
        "y_true": y_true,
        filename.split('_')[0] + "_pred": y_pred
    })
    df.to_csv(os.path.join(out_dir, filename), index=False)

def save_cv_metrics_json(metrics_dict, out_path):
    with open(out_path, "w") as f:
        json.dump(metrics_dict, f, indent=2)

def save_model(model, models_dir, filename):
    import joblib
    joblib.dump(model, os.path.join(models_dir, filename))

def cross_val_with_metrics(model_class, model_params, X_trainval, y_trainval, tscv, batch_size, device, seq_len, epochs, grad_clip, early_stopping_patience):
    mse, mae, dir_acc, prec, rec, f1 = [], [], [], [], [], []
    fold_iter = tqdm(list(tscv.split(X_trainval)), desc="Cross-Validation Folds", dynamic_ncols=True)
    for train_idx, val_idx in fold_iter:
        X_tr, X_val = X_trainval[train_idx], X_trainval[val_idx]
        y_tr, y_val = y_trainval[train_idx], y_trainval[val_idx]
        train_loader, val_loader = get_dataloaders(X_tr, y_tr, X_val, y_val, batch_size, device, seq_len=seq_len)
        model = model_class(**model_params).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=model_params.get("lr", 0.001),
            weight_decay=model_params.get("weight_decay", 1e-5)  # L2 regularization
        )
        criterion = torch.nn.SmoothL1Loss(beta=0.1)  # More robust loss for noisy data
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 'min', patience=3, factor=0.1  # More aggressive schedule
        )
        model, _ = train_full_model(
            model, train_loader, val_loader, optimizer, criterion, scheduler, device,
            epochs=epochs, grad_clip=grad_clip, early_stopping_patience=early_stopping_patience, save_path=None
        )
        val_loss, val_preds, val_targets = validate_one_epoch(model, val_loader, criterion, device)
        mse.append(mean_squared_error(val_targets, val_preds))
        mae.append(mean_absolute_error(val_targets, val_preds))
        d_pred, d_true = np.sign(val_preds), np.sign(val_targets)
        dir_acc.append(np.mean(d_pred == d_true))
        prec.append(precision_score(d_true, d_pred, labels=[-1, 1], average='macro', zero_division=0))
        rec.append(recall_score(d_true, d_pred, labels=[-1, 1], average='macro', zero_division=0))
        f1.append(f1_score(d_true, d_pred, labels=[-1, 1], average='macro', zero_division=0))
    results = {
        "mse": mse,
        "mae": mae,
        "dir_acc": dir_acc,
        "precision": prec,
        "recall": rec,
        "f1": f1
    }
    return results

# --------------------------------------------------------------------------
# Hyperparameter tuning with Optuna 
def objective(trial, model_class, X_train, y_train, X_val, y_val, device, seq_len=60):
    hidden_dim = trial.suggest_categorical("hidden_dim", [128, 256, 512])
    num_layers = trial.suggest_int("num_layers", 2, 4)
    dropout = trial.suggest_float("dropout", 0.2, 0.4)
    lr = trial.suggest_loguniform("lr", 1e-5, 1e-2)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    grad_clip = trial.suggest_float("grad_clip", 1.0, 2.0)  # Reasonable range
    bidirectional = False  # Always unidirectional for time series
    weight_decay = trial.suggest_loguniform("weight_decay", 1e-6, 1e-3)  # L2 regularization

    train_loader, val_loader = get_dataloaders(X_train, y_train, X_val, y_val, batch_size, device, seq_len=seq_len)
    model = model_class(
        input_dim=X_train.shape[1],
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        bidirectional=bidirectional
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )
    criterion = torch.nn.SmoothL1Loss(beta=0.1)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.1)
    best_val_loss = float('inf')
    patience_counter = 0
    for epoch in range(80):
        train_one_epoch(model, train_loader, optimizer, criterion, device, grad_clip)
        val_loss, _, _ = validate_one_epoch(model, val_loader, criterion, device)
        scheduler.step(val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 5:
                break
    return best_val_loss

def tune_hyperparameters(model_class, X_train, y_train, X_val, y_val, device, seq_len=60, n_trials=30):
    trials_iter = tqdm(range(n_trials), desc="Optuna Trials", dynamic_ncols=True)
    results = []
    def wrapped_objective(trial):
        result = objective(trial, model_class, X_train, y_train, X_val, y_val, device, seq_len)
        trials_iter.update(1)
        results.append(result)
        return result
    study = optuna.create_study(direction="minimize")
    study.optimize(wrapped_objective, n_trials=n_trials)
    trials_iter.close()
    print("Best trial:", study.best_trial.params)
    return study.best_trial.params
