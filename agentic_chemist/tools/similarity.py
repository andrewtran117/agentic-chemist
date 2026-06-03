"""compute_similarity: Morgan-fingerprint Tanimoto similarity between two molecules."""

from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs


def compute_similarity(smiles_a: str, smiles_b: str, radius: int = 2, n_bits: int = 2048) -> float | None:
    """Compute Tanimoto similarity between two molecules using Morgan fingerprints.

    Returns a float in [0, 1], or None if either SMILES is invalid.
    """
    mol_a = Chem.MolFromSmiles(smiles_a)
    mol_b = Chem.MolFromSmiles(smiles_b)

    if mol_a is None or mol_b is None:
        return None

    fp_a = AllChem.GetMorganFingerprintAsBitVect(mol_a, radius, nBits=n_bits)
    fp_b = AllChem.GetMorganFingerprintAsBitVect(mol_b, radius, nBits=n_bits)

    return DataStructs.TanimotoSimilarity(fp_a, fp_b)
