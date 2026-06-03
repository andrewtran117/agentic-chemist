"""parse_molecule: canonicalize SMILES and compute key descriptors via RDKit."""

from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, QED, Lipinski

from agentic_chemist.schemas import MoleculeInfo


def parse_molecule(smiles: str) -> MoleculeInfo:
    """Parse a SMILES string and return molecular descriptors.

    Canonicalizes the input, validates it, and computes physicochemical
    properties used for drug-likeness triage.
    """
    if not smiles or not smiles.strip():
        return MoleculeInfo(
            smiles=smiles,
            canonical_smiles="",
            is_valid=False,
        )

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return MoleculeInfo(
            smiles=smiles,
            canonical_smiles="",
            is_valid=False,
        )

    canonical = Chem.MolToSmiles(mol)

    mw = Descriptors.ExactMolWt(mol)
    logp = Descriptors.MolLogP(mol)
    hbd = rdMolDescriptors.CalcNumHBD(mol)
    hba = rdMolDescriptors.CalcNumHBA(mol)
    tpsa = Descriptors.TPSA(mol)
    rotatable = rdMolDescriptors.CalcNumRotatableBonds(mol)
    qed_score = QED.qed(mol)
    formula = rdMolDescriptors.CalcMolFormula(mol)
    num_rings = rdMolDescriptors.CalcNumRings(mol)

    # Lipinski's Rule of Five
    violations = sum([
        mw > 500,
        logp > 5,
        hbd > 5,
        hba > 10,
    ])

    return MoleculeInfo(
        smiles=smiles,
        canonical_smiles=canonical,
        is_valid=True,
        molecular_weight=round(mw, 2),
        logp=round(logp, 2),
        hbd=hbd,
        hba=hba,
        tpsa=round(tpsa, 2),
        rotatable_bonds=rotatable,
        qed=round(qed_score, 3),
        lipinski_pass=violations == 0,
        lipinski_violations=violations,
        formula=formula,
        num_rings=num_rings,
    )
