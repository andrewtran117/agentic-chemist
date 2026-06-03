"""predict_admet: predict ADMET endpoints with applicability domain checks."""

import pickle
from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs

from agentic_chemist.schemas import Prediction

MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "models"
FP_RADIUS = 2
FP_NBITS = 2048

# Applicability domain thresholds (Tanimoto to nearest training molecule)
AD_IN_DOMAIN = 0.3
AD_BORDERLINE = 0.2

ENDPOINTS = {
    "solubility": {
        "model_file": "solubility_rf.pkl",
        "train_fps_file": "solubility_train_fps.npy",
        "model_id": "rf_solubility_aqsoldb_scaffold",
        "units": "logS",
        "task": "regression",
    },
    "herg": {
        "model_file": "herg_rf.pkl",
        "train_fps_file": "herg_train_fps.npy",
        "model_id": "rf_herg_scaffold",
        "units": "probability",
        "task": "classification",
    },
}

# Cache loaded models
_model_cache: dict = {}


def _load_model(endpoint: str):
    """Load model and training fingerprints, with caching."""
    if endpoint in _model_cache:
        return _model_cache[endpoint]

    config = ENDPOINTS[endpoint]
    model_path = MODELS_DIR / config["model_file"]
    fps_path = MODELS_DIR / config["train_fps_file"]

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. Run train_admet.py first."
        )

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    train_fps = np.load(fps_path)

    # Pre-compute RDKit DataStructs fingerprints for fast Tanimoto
    train_bvs = []
    for row in train_fps:
        bv = DataStructs.ExplicitBitVect(FP_NBITS)
        on_bits = np.where(row == 1)[0]
        for bit in on_bits:
            bv.SetBit(int(bit))
        train_bvs.append(bv)

    _model_cache[endpoint] = (model, train_fps, train_bvs, config)
    return model, train_fps, train_bvs, config


def _nearest_tanimoto(fp_bv, train_bvs) -> float:
    """Find the maximum Tanimoto similarity to any training molecule."""
    if not train_bvs:
        return 0.0
    sims = DataStructs.BulkTanimotoSimilarity(fp_bv, train_bvs)
    return max(sims)


def _applicability_flag(tanimoto: float) -> str:
    """Classify applicability based on nearest-neighbor Tanimoto."""
    if tanimoto >= AD_IN_DOMAIN:
        return "in_domain"
    elif tanimoto >= AD_BORDERLINE:
        return "borderline"
    else:
        return "out_of_domain"


def predict_admet(smiles: str, endpoint: str) -> Prediction | None:
    """Predict an ADMET endpoint for a molecule.

    Args:
        smiles: SMILES string of the query molecule.
        endpoint: one of "solubility", "herg".

    Returns:
        Prediction with value, units, model ID, and applicability flag.
        None if the SMILES is invalid or the endpoint is unknown.
    """
    if endpoint not in ENDPOINTS:
        raise ValueError(f"Unknown endpoint '{endpoint}'. Available: {list(ENDPOINTS.keys())}")

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    model, train_fps, train_bvs, config = _load_model(endpoint)

    # Compute fingerprint
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, FP_RADIUS, nBits=FP_NBITS)
    fp_arr = np.array(fp).reshape(1, -1)

    # Predict
    if config["task"] == "classification":
        value = float(model.predict_proba(fp_arr)[0, 1])
    else:
        value = float(model.predict(fp_arr)[0])

    # Applicability domain
    nearest_tan = _nearest_tanimoto(fp, train_bvs)

    return Prediction(
        smiles=smiles,
        endpoint=endpoint,
        value=round(value, 4),
        units=config["units"],
        model_id=config["model_id"],
        applicability_flag=_applicability_flag(nearest_tan),
        nearest_train_tanimoto=round(nearest_tan, 4),
    )
