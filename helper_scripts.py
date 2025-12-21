import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def evaluate_model(name, model, X_tr, y_tr, X_te, y_te):
    """
    Train model, evaluate on test set, and perform cross-validation
    """
    # Train the model
    model.fit(X_tr, y_tr)
    
    # Test set predictions
    pred = model.predict(X_te)
    
    # Cross-validation on training set (5-fold)
    print(f"\n{'='*60}")
    print(f"Cross-Validation: {name}")
    print(f"{'='*60}")
    
    cv_r2 = cross_val_score(model, X_tr, y_tr, cv=5, scoring='r2', n_jobs=-1)
    cv_mae = -cross_val_score(model, X_tr, y_tr, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
    cv_rmse = np.sqrt(-cross_val_score(model, X_tr, y_tr, cv=5, scoring='neg_mean_squared_error', n_jobs=-1))
    
    print(f"  R² scores:   {cv_r2}")
    print(f"  Mean R²:     {cv_r2.mean():.4f} (+/- {cv_r2.std():.4f})")
    print(f"  MAE scores:  {cv_mae}")
    print(f"  Mean MAE:    €{cv_mae.mean():,.2f} (+/- €{cv_mae.std():,.2f})")
    print(f"  RMSE scores: {cv_rmse}")
    print(f"  Mean RMSE:   €{cv_rmse.mean():,.2f} (+/- €{cv_rmse.std():,.2f})")
    
    return {
        "Model": name,
        "MAE": mean_absolute_error(y_te, pred),
        "RMSE": np.sqrt(mean_squared_error(y_te, pred)),
        "R2": r2_score(y_te, pred),
        "CV_R2_Mean": cv_r2.mean(),
        "CV_R2_Std": cv_r2.std(),
        "CV_MAE_Mean": cv_mae.mean(),
        "CV_RMSE_Mean": cv_rmse.mean()
    }

# Use MAE (negative in sklearn scoring conventions)
def cv_mae(model):
    scores = cross_val_score(
        model, X_cv, y_cv,
        cv=tscv,
        scoring="neg_mean_absolute_error",
        n_jobs=-1
    )
    return -scores.mean(), scores.std()

def remove_duplicates(df):
    """
    Checking dataframe for duplicates and removing if present    

    NOTICE: Currently Not Formatting As Desired
    """
    print(f"Total rows: {len(df)}")
    print(f"Duplicate rows: {df.duplicated().sum()}")
    if df.duplicated().sum() > 0:
        print(f"Removing {df.duplicated().sum()} duplicate rows...")
        df = df.drop_duplicates()
        print(f"Rows after removing duplicates: {len(df)}")
    else:
        print("No duplicate rows found.")

