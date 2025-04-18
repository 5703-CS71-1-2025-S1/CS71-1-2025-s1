# data_loader.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_credit_data(test_size=0.2, val_size=0.2, random_state=42):
    # Load UCI dataset
    df = pd.read_excel("default of credit card clients.xls", header=1)

    # Drop ID column and rename target
    df = df.drop(columns=["ID"])
    df = df.rename(columns={"default payment next month": "target"})

    # Split features and target
    X = df.drop(columns=["target"])
    y = df["target"]

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split into train, validation, test sets (60/20/20)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_scaled, y, test_size=val_size + test_size, random_state=random_state, stratify=y
    )
    val_ratio_adjusted = val_size / (val_size + test_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=1 - val_ratio_adjusted, random_state=random_state, stratify=y_temp
    )

    return (X_train, y_train), (X_val, y_val), (X_test, y_test), scaler

if __name__ == "__main__":
    (X_train, y_train), (X_val, y_val), (X_test, y_test), scaler = load_credit_data()
    print(f"Train: {X_train.shape}, Validation: {X_val.shape}, Test: {X_test.shape}")
