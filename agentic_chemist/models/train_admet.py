"""Train ADMET models (solubility + hERG) with scaffold splits.

Produces:
  - Trained sklearn models saved to models/
  - Training set fingerprints for applicability domain checks
  - A training report with random-vs-scaffold split performance gap
"""

import json
import pickle
from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    roc_auc_score,
    matthews_corrcoef,
)
from tdc.single_pred import ADME, Tox

MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "models"
FP_RADIUS = 2
FP_NBITS = 2048


def smiles_to_fp(smiles: str) -> np.ndarray | None:
    """Convert SMILES to a Morgan fingerprint bit vector."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, FP_RADIUS, nBits=FP_NBITS)
    return np.array(fp)


def prepare_data(df):
    """Convert a TDC dataframe to fingerprint matrix + labels, dropping invalid SMILES."""
    fps, labels, valid_smiles = [], [], []
    for _, row in df.iterrows():
        fp = smiles_to_fp(row["Drug"])
        if fp is not None:
            fps.append(fp)
            labels.append(row["Y"])
            valid_smiles.append(row["Drug"])
    return np.array(fps), np.array(labels), valid_smiles


def train_solubility():
    """Train solubility regressor on AqSolDB with scaffold split."""
    print("=== Solubility (AqSolDB) ===")
    data = ADME(name="Solubility_AqSolDB")

    # Scaffold split
    scaffold_split = data.get_split(method="scaffold")
    X_train, y_train, train_smiles = prepare_data(scaffold_split["train"])
    X_val, y_val, _ = prepare_data(scaffold_split["valid"])
    X_test, y_test, _ = prepare_data(scaffold_split["test"])

    model = RandomForestRegressor(n_estimators=500, n_jobs=-1, random_state=42)
    model.fit(X_train, y_train)

    # Scaffold split metrics
    y_pred_test = model.predict(X_test)
    scaffold_metrics = {
        "mae": round(mean_absolute_error(y_test, y_pred_test), 4),
        "rmse": round(np.sqrt(mean_squared_error(y_test, y_pred_test)), 4),
        "r2": round(r2_score(y_test, y_pred_test), 4),
    }
    print(f"  Scaffold split — MAE: {scaffold_metrics['mae']}, RMSE: {scaffold_metrics['rmse']}, R²: {scaffold_metrics['r2']}")

    # Random split for comparison
    random_split = data.get_split(method="random", seed=42)
    X_train_r, y_train_r, _ = prepare_data(random_split["train"])
    X_test_r, y_test_r, _ = prepare_data(random_split["test"])

    model_random = RandomForestRegressor(n_estimators=500, n_jobs=-1, random_state=42)
    model_random.fit(X_train_r, y_train_r)
    y_pred_r = model_random.predict(X_test_r)
    random_metrics = {
        "mae": round(mean_absolute_error(y_test_r, y_pred_r), 4),
        "rmse": round(np.sqrt(mean_squared_error(y_test_r, y_pred_r)), 4),
        "r2": round(r2_score(y_test_r, y_pred_r), 4),
    }
    print(f"  Random split   — MAE: {random_metrics['mae']}, RMSE: {random_metrics['rmse']}, R²: {random_metrics['r2']}")
    print(f"  Gap (scaffold - random MAE): {scaffold_metrics['mae'] - random_metrics['mae']:.4f}")

    # Save model + training fingerprints (for applicability domain)
    MODELS_DIR.mkdir(exist_ok=True)
    with open(MODELS_DIR / "solubility_rf.pkl", "wb") as f:
        pickle.dump(model, f)
    np.save(MODELS_DIR / "solubility_train_fps.npy", X_train)

    report = {
        "endpoint": "solubility",
        "dataset": "AqSolDB",
        "model": "RandomForestRegressor",
        "n_train": len(X_train),
        "n_test": len(X_test),
        "scaffold_metrics": scaffold_metrics,
        "random_metrics": random_metrics,
        "split_gap_mae": round(scaffold_metrics["mae"] - random_metrics["mae"], 4),
    }
    with open(MODELS_DIR / "solubility_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print(f"  Saved to {MODELS_DIR}/solubility_rf.pkl")
    return report


def train_herg():
    """Train hERG blocker classifier with scaffold split."""
    print("\n=== hERG ===")
    data = Tox(name="hERG")

    scaffold_split = data.get_split(method="scaffold")
    X_train, y_train, train_smiles = prepare_data(scaffold_split["train"])
    X_val, y_val, _ = prepare_data(scaffold_split["valid"])
    X_test, y_test, _ = prepare_data(scaffold_split["test"])

    model = RandomForestClassifier(n_estimators=500, n_jobs=-1, random_state=42)
    model.fit(X_train, y_train)

    # Scaffold split metrics
    y_pred_test = model.predict(X_test)
    y_prob_test = model.predict_proba(X_test)[:, 1]
    scaffold_metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred_test), 4),
        "auroc": round(roc_auc_score(y_test, y_prob_test), 4),
        "mcc": round(matthews_corrcoef(y_test, y_pred_test), 4),
    }
    print(f"  Scaffold split — Acc: {scaffold_metrics['accuracy']}, AUROC: {scaffold_metrics['auroc']}, MCC: {scaffold_metrics['mcc']}")

    # Random split for comparison
    random_split = data.get_split(method="random", seed=42)
    X_train_r, y_train_r, _ = prepare_data(random_split["train"])
    X_test_r, y_test_r, _ = prepare_data(random_split["test"])

    model_random = RandomForestClassifier(n_estimators=500, n_jobs=-1, random_state=42)
    model_random.fit(X_train_r, y_train_r)
    y_pred_r = model_random.predict(X_test_r)
    y_prob_r = model_random.predict_proba(X_test_r)[:, 1]
    random_metrics = {
        "accuracy": round(accuracy_score(y_test_r, y_pred_r), 4),
        "auroc": round(roc_auc_score(y_test_r, y_prob_r), 4),
        "mcc": round(matthews_corrcoef(y_test_r, y_pred_r), 4),
    }
    print(f"  Random split   — Acc: {random_metrics['accuracy']}, AUROC: {random_metrics['auroc']}, MCC: {random_metrics['mcc']}")
    print(f"  Gap (scaffold - random AUROC): {scaffold_metrics['auroc'] - random_metrics['auroc']:.4f}")

    # Save
    with open(MODELS_DIR / "herg_rf.pkl", "wb") as f:
        pickle.dump(model, f)
    np.save(MODELS_DIR / "herg_train_fps.npy", X_train)

    report = {
        "endpoint": "herg",
        "dataset": "hERG",
        "model": "RandomForestClassifier",
        "n_train": len(X_train),
        "n_test": len(X_test),
        "scaffold_metrics": scaffold_metrics,
        "random_metrics": random_metrics,
        "split_gap_auroc": round(scaffold_metrics["auroc"] - random_metrics["auroc"], 4),
    }
    with open(MODELS_DIR / "herg_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print(f"  Saved to {MODELS_DIR}/herg_rf.pkl")
    return report


if __name__ == "__main__":
    sol_report = train_solubility()
    herg_report = train_herg()
    print("\nDone. Reports saved to models/")
