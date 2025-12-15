# ---- Script to create functions used during modelling a lot -------------
# -------------------------------------------------------------------------
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, precision_score, recall_score, f1_score
import json
import joblib
import os
import torch

# -------------------------------------------------------------------------
# Set configuerations for the deep learning models
def config():
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    EPOCHS = 100
    SEED = 123
    return DEVICE, EPOCHS, SEED
# ------------------------------------------------------------------------

# load and prep data function
def load_and_prep_data(drop_date=True):
    # always use the same path for all models
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'final_data.csv')
    data = pd.read_csv(data_path)
    print_data_summary(data)
    drop_cols = ['Daily_Return']
    if drop_date and 'Date' in data.columns:
        drop_cols.append('Date')
    X = data.drop(columns=drop_cols)
    y = data['Daily_Return'].shift(-1) # predict next day's return
    X, y = X.iloc[:-1], y.iloc[:-1] # drop last row with NaN target
    return X, y

# -------------------------------------------------------------------------
# print data summary
def print_data_summary(df):
    # print a summary of the dataframe: shape, columns, missing values
    print("Data Summary:")
    print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    print("Columns:", list(df.columns))
    print("Missing values per column:")
    print(df.isnull().sum())
    print("-" * 40)
    
# -------------------------------------------------------------------------
# # run the cross validation to get metrics for both models
def get_cv_scores(model_cls, model_params, X_trainval, y_trainval, tscv):
   
    mse, mae, dir_acc, prec, rec, f1 = [], [], [], [], [], [] # lists to store metrics for each fold
    for train_idx, val_idx in tscv.split(X_trainval): # loop through the time series splits

        X_tr, X_val = X_trainval.iloc[train_idx], X_trainval.iloc[val_idx] # get the train and validation sets
        y_tr, y_val = y_trainval.iloc[train_idx], y_trainval.iloc[val_idx] # get the train and validation targets

        scaler_cv = StandardScaler() # define a scaler for this fold
        X_tr_scaled = scaler_cv.fit_transform(X_tr) # fit and transform the train set
        X_val_scaled = scaler_cv.transform(X_val) # transform the validation set

        params = model_params if model_params is not None else {} # get model parameters
        model = model_cls(**params) # instantiate the model
        model.fit(X_tr_scaled, y_tr) # fit the model
        preds = model.predict(X_val_scaled) # make predictions

        # calculate and store metrics
        mse.append(mean_squared_error(y_val, preds)) # MSE
        mae.append(mean_absolute_error(y_val, preds)) # MAE
        d_pred, d_true = np.sign(preds), np.sign(y_val) # directional predictions and true values
        dir_acc.append(np.mean(d_pred == d_true)) # directional accuracy
        prec.append(precision_score(d_true, d_pred, labels=[-1, 1], average='macro', zero_division=0)) # precision
        rec.append(recall_score(d_true, d_pred, labels=[-1, 1], average='macro', zero_division=0)) # recall
        f1.append(f1_score(d_true, d_pred, labels=[-1, 1], average='macro', zero_division=0)) # F1 score
    
    def mean_ci(metric_list): # define a function to calculate mean and 95% CI
        arr = np.array(metric_list) # convert to numpy array

        mean = np.mean(arr) # calculate mean
        std = np.std(arr, ddof=1) # sample standard deviation

        n = len(arr) # get the number of samples
        ci95 = 1.96 * std / np.sqrt(n) if n > 1 else 0 # 95% CI calculation
        return float(mean), float(ci95) # return as floats

    results = {} # dictionary to store results
    for name, values in zip( # loop through metric names and their values
        ['mse', 'mae', 'dir_acc', 'precision', 'recall', 'f1'],
        [mse, mae, dir_acc, prec, rec, f1]
    ):
        mean, ci = mean_ci(values) # calculate mean and CI
        results[name] = values # store all fold values
        results[name + "_mean"] = mean # store mean
        results[name + "_ci95"] = ci # store 95% CI

    return results # return the results dictionary
# -------------------------------------------------------------------------
# save CV metrics as JSON
def save_cv_metrics_json(metrics_dict, out_path):
    # save the cross-validation metrics dictionary as a JSON file
    with open(out_path, "w") as f:
        json.dump(metrics_dict, f, indent=2)

# -------------------------------------------------------------------------
# save test predictions as CSV
def save_test_predictions(y_true, y_pred, out_dir, filename):
    # save the test predictions as a CSV file
    df = pd.DataFrame({
        "y_true": y_true,
        filename.split('_')[0] + "_pred": y_pred
    })
    df.to_csv(os.path.join(out_dir, filename), index=False)

# -------------------------------------------------------------------------
# save model
def save_model(model, models_dir, filename):
    # save the trained model using joblib
    joblib.dump(model, os.path.join(models_dir, filename))

# -------------------------------------------------------------------------
# directional metrics function
def directional_metrics(y_true, y_pred):
    d_pred, d_true = np.sign(y_pred), np.sign(y_true)
    dir_acc = float(np.mean(d_pred == d_true))
    precision = float(precision_score(d_true, d_pred, labels=[-1, 1], average='macro', zero_division=0))
    recall = float(recall_score(d_true, d_pred, labels=[-1, 1], average='macro', zero_division=0))
    f1 = float(f1_score(d_true, d_pred, labels=[-1, 1], average='macro', zero_division=0))
    return dir_acc, precision, recall, f1