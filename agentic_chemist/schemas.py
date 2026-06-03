from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class MoleculeInfo:
    smiles: str
    canonical_smiles: str
    is_valid: bool
    molecular_weight: Optional[float] = None
    logp: Optional[float] = None
    hbd: Optional[int] = None  # hydrogen bond donors
    hba: Optional[int] = None  # hydrogen bond acceptors
    tpsa: Optional[float] = None  # topological polar surface area
    rotatable_bonds: Optional[int] = None
    qed: Optional[float] = None  # quantitative estimate of drug-likeness
    lipinski_pass: Optional[bool] = None
    lipinski_violations: Optional[int] = None
    formula: Optional[str] = None
    num_rings: Optional[int] = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Prediction:
    smiles: str
    endpoint: str
    value: float
    units: str
    model_id: str
    applicability_flag: str  # "in_domain", "borderline", "out_of_domain"
    nearest_train_tanimoto: float

    def to_dict(self) -> dict:
        return asdict(self)
