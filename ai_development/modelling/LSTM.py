# -- Script running the LSTM model and saving cv results and test preds --
# ------------------------------------------------------------------------
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from utils import load_and_prep_data, config
from sklearn.preprocessing import StandardScaler
import os
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm
import optuna
from training import (
    train_full_model,
    evaluate_model,
    save_test_predictions,
    save_cv_metrics_json,
    cross_val_with_metrics,
    validate_one_epoch
)
import warnings
warnings.filterwarnings("ignore")

# ------------------------------------------------------------------------
# Device configuration
DEVICE,EPOCHS,  SEED = config()

# ------------------------------------------------------------------------
# Load the final dataset
X, y = load_and_prep_data(drop_date=True)
test_size = int(0.2 * len(X))
X_trainval, X_test = X.iloc[:-test_size], X.iloc[-test_size:]
y_trainval, y_test = y.iloc[:-test_size], y.iloc[-test_size:]

# Scale features
scaler = StandardScaler()
X_trainval_scaled = scaler.fit_transform(X_trainval)
X_test_scaled = scaler.transform(X_test)

# ------------------------------------------------------------------------
# LSTM with Attention
class FinancialLSTM(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=2, output_dim=1, dropout=0.3, bidirectional=False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.lstm = torch.nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        lstm_out_dim = hidden_dim * (2 if bidirectional else 1)
        self.fc = torch.nn.Linear(lstm_out_dim, output_dim)
        self.dropout = torch.nn.Dropout(dropout)
        self.layer_norm = torch.nn.LayerNorm(lstm_out_dim)
        self.attn = torch.nn.Linear(lstm_out_dim, 1)
        # Weight initialization
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                torch.nn.init.orthogonal_(param)
            elif 'bias' in name:
                torch.nn.init.constant_(param, 0)
        torch.nn.init.xavier_uniform_(self.fc.weight)
        torch.nn.init.constant_(self.fc.bias, 0)
        torch.nn.init.xavier_uniform_(self.attn.weight)
        torch.nn.init.constant_(self.attn.bias, 0)

    def forward(self, x):
        h0 = torch.zeros(
            self.num_layers * (2 if self.bidirectional else 1),
            x.size(0),
            self.hidden_dim,
            device=x.device
        )
        c0 = torch.zeros(
            self.num_layers * (2 if self.bidirectional else 1),
            x.size(0),
            self.hidden_dim,
            device=x.device
        )
        out, _ = self.lstm(x, (h0, c0))  # (batch, seq_len, lstm_out_dim)
        out = self.layer_norm(out)
        out = self.dropout(out)
        attn_weights = torch.softmax(self.attn(out).squeeze(-1), dim=1)  # (batch, seq_len)
        context = torch.sum(out * attn_weights.unsqueeze(-1), dim=1)     # (batch, lstm_out_dim)
        out = self.fc(context)
        return out

# Set random seeds for reproducibility
torch.manual_seed(SEED)
np.random.seed(SEED)

# Sequence/data loader helpers
def create_sequences(X, y, seq_len=1):
    Xs, ys = [], []
    for i in range(len(X) - seq_len + 1):
        Xs.append(X[i:i+seq_len])
        ys.append(y[i+seq_len-1])
    return np.array(Xs), np.array(ys)

SEQ_LEN = 60

def get_sequence_loader(X, y, seq_len, batch_size):
    X_seq, y_seq = create_sequences(X, y, seq_len)
    return DataLoader(
        TensorDataset(
            torch.tensor(X_seq, dtype=torch.float32),
            torch.tensor(y_seq, dtype=torch.float32)
        ),
        batch_size=batch_size,
        shuffle=False
    )

# ------------------------------------------------------------------------
# Cross-validation on trainval set (time series split)
tscv = TimeSeriesSplit(n_splits=5)

# --- Optuna hyperparameter tuning on the last (most recent) fold only ---

def last_fold_optuna_objective(trial, model_class, X_trainval_scaled, y_trainval, tscv, device, seq_len, epochs=20):
    split_indices = list(tscv.split(X_trainval_scaled))
    if not split_indices:
        raise ValueError("No folds found in TimeSeriesSplit. Check your data and n_splits.")
    train_idx, val_idx = split_indices[-1]
    X_train, X_val = X_trainval_scaled[train_idx], X_trainval_scaled[val_idx]
    y_train, y_val = y_trainval[train_idx], y_trainval[val_idx]
    # Hyperparameters to tune
    hidden_dim = trial.suggest_categorical("hidden_dim", [128, 256, 512])
    num_layers = trial.suggest_int("num_layers", 2, 4)
    dropout = trial.suggest_float("dropout", 0.2, 0.4)
    lr = trial.suggest_loguniform("lr", 1e-5, 1e-2)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    grad_clip = trial.suggest_float("grad_clip", 1.0, 2.0)
    bidirectional = False
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3)
    # Print hyperparameters for this trial
    print(f"\nOptuna trial {trial.number}: hidden_dim={hidden_dim}, num_layers={num_layers}, dropout={dropout:.3f}, lr={lr:.5f}, batch_size={batch_size}, grad_clip={grad_clip:.2f}, weight_decay={weight_decay:.6f}")
    # Data loaders
    train_loader = get_sequence_loader(X_train, y_train, seq_len, batch_size)
    val_loader = get_sequence_loader(X_val, y_val, seq_len, batch_size)
    # Model
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
    # Train model
    orig_tqdm = tqdm
    try:
        globals()['tqdm'] = lambda *a, **k: (x for x in range(a[0])) if a else iter([])
        model, _ = train_full_model(
            model, train_loader, val_loader, optimizer, criterion, scheduler, device,
            epochs=epochs, grad_clip=grad_clip, early_stopping_patience=5, save_path=None
        )
    finally:
        globals()['tqdm'] = orig_tqdm
    val_loss, _, _ = validate_one_epoch(model, val_loader, criterion, device)
    return val_loss

def tune_hyperparameters_last_fold(model_class, X_trainval_scaled, y_trainval, tscv, device, seq_len=60, n_trials=30, epochs=20):
    print(f"Running Optuna hyperparameter tuning on last fold ({n_trials} trials, {epochs} epochs per trial)...")
    study = optuna.create_study(direction="minimize")
    # Use Optuna's default progress bar 
    def wrapped_objective(trial):
        return last_fold_optuna_objective(
            trial, model_class, X_trainval_scaled, y_trainval, tscv, device, seq_len, epochs
        )
    study.optimize(wrapped_objective, n_trials=n_trials, show_progress_bar=True)
    print("Best trial:", study.best_trial.params)
    return study.best_trial.params

# --- Use Optuna to find best hyperparameters on last fold ---
best_params = tune_hyperparameters_last_fold(
    FinancialLSTM,
    X_trainval_scaled,
    y_trainval.values,
    tscv,
    DEVICE,
    seq_len=SEQ_LEN,
    n_trials=30,
    epochs=80
)
print("Best hyperparameters found by Optuna:", best_params)

# --- Use best hyperparameters for cross-validation ---
model_params = {
    "input_dim": X.shape[1],
    "hidden_dim": best_params["hidden_dim"],
    "num_layers": best_params["num_layers"],
    "dropout": best_params["dropout"],
    "bidirectional": best_params["bidirectional"]
}
cv_metrics = cross_val_with_metrics(
    FinancialLSTM,
    model_params,
    X_trainval_scaled,
    y_trainval.values,
    tscv,
    best_params["batch_size"],
    DEVICE,
    SEQ_LEN,
    EPOCHS,
    grad_clip=best_params["grad_clip"],
    early_stopping_patience=10
)

print("CV MSEs:", cv_metrics["mse"])
print("CV MAEs:", cv_metrics["mae"])
print("CV Directional Accuracies:", cv_metrics["dir_acc"])
print("CV Precisions:", cv_metrics["precision"])
print("CV Recalls:", cv_metrics["recall"])
print("CV F1s:", cv_metrics["f1"])
print("Mean CV MSE:", np.mean(cv_metrics["mse"]))
print("Mean CV MAE:", np.mean(cv_metrics["mae"]))
print("Mean CV Dir Acc:", np.mean(cv_metrics["dir_acc"]))
print("Mean CV Precision:", np.mean(cv_metrics["precision"]))
print("Mean CV Recall:", np.mean(cv_metrics["recall"]))
print("Mean CV F1:", np.mean(cv_metrics["f1"]))

# Save CV metrics
cv_dir = os.path.join(os.path.dirname(__file__), '..', 'results_analysis', 'results', 'cv_results')
os.makedirs(cv_dir, exist_ok=True)
save_cv_metrics_json(cv_metrics, os.path.join(cv_dir, "lstm_cv_metrics.json"))

# ------------------------------------------------------------------------
# Full out-of-sample test predictions

test_loader = get_sequence_loader(X_test_scaled, y_test.values, SEQ_LEN, best_params["batch_size"])
trainval_loader = get_sequence_loader(X_trainval_scaled, y_trainval.values, SEQ_LEN, best_params["batch_size"])

models_dir = os.path.join(os.path.dirname(__file__), 'saved_models')
os.makedirs(models_dir, exist_ok=True)

model = FinancialLSTM(
    input_dim=X.shape[1],
    hidden_dim=best_params["hidden_dim"],
    num_layers=best_params["num_layers"],
    dropout=best_params["dropout"],
    bidirectional=best_params["bidirectional"]
).to(DEVICE)
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=best_params["lr"],
    weight_decay=best_params.get("weight_decay", 1e-5)
)
criterion = torch.nn.SmoothL1Loss(beta=0.1)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 'min', patience=3, factor=0.1
)

model, history = train_full_model(
    model, trainval_loader, None, optimizer, criterion, scheduler, DEVICE,
    epochs=EPOCHS, grad_clip=best_params["grad_clip"], early_stopping_patience=10, save_path=os.path.join(models_dir, "lstm_final.pt")
)

torch.save(model.state_dict(), os.path.join(models_dir, "lstm_final.pt"))

results = evaluate_model(model, test_loader, DEVICE)
print("Out-of-sample test metrics:", results["test_metrics"])

out_dir = os.path.join(os.path.dirname(__file__), '..', 'results_analysis', 'results', 'out_of_sample')
os.makedirs(out_dir, exist_ok=True)
save_test_predictions(
    np.array([y for _, y in test_loader.dataset]),
    results["predicted_returns"],
    out_dir,
    "lstm_test_predictions.csv"
)