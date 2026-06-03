"""Tests for parse_molecule tool."""

import pytest
from agentic_chemist.tools.parse import parse_molecule


class TestParseValid:
    """Test parsing of known valid molecules with verifiable properties."""

    def test_aspirin(self):
        result = parse_molecule("CC(=O)Oc1ccccc1C(=O)O")
        assert result.is_valid
        assert result.formula == "C9H8O4"
        assert result.lipinski_pass is True
        assert result.hbd == 1
        assert result.hba == 3  # RDKit HBA excludes ester oxygen
        assert result.num_rings == 1
        assert 170 < result.molecular_weight < 190  # MW ~180.16

    def test_caffeine(self):
        result = parse_molecule("Cn1c(=O)c2c(ncn2C)n(C)c1=O")
        assert result.is_valid
        assert result.formula == "C8H10N4O2"
        assert result.lipinski_pass is True
        assert result.hbd == 0  # no OH or NH
        assert 190 < result.molecular_weight < 200  # MW ~194.19

    def test_ethanol(self):
        result = parse_molecule("CCO")
        assert result.is_valid
        assert result.formula == "C2H6O"
        assert result.lipinski_pass is True
        assert result.num_rings == 0

    def test_canonical_smiles_normalization(self):
        """Different SMILES for same molecule should canonicalize to the same string."""
        r1 = parse_molecule("c1ccccc1")  # benzene
        r2 = parse_molecule("C1=CC=CC=C1")  # benzene alternate
        assert r1.canonical_smiles == r2.canonical_smiles

    def test_qed_range(self):
        result = parse_molecule("CC(=O)Oc1ccccc1C(=O)O")  # aspirin
        assert 0.0 <= result.qed <= 1.0


class TestParseLipinski:
    """Test Lipinski rule-of-five violation detection."""

    def test_lipinski_violator(self):
        # Cyclosporine A (large cyclic peptide) — known Lipinski violator
        # Use a simpler known violator: MW > 500, logP > 5
        # This is a big greasy molecule
        big_mol = "CCCCCCCCCCCCCCCCCCCCCCCCCC(=O)OCC"  # long-chain ester
        result = parse_molecule(big_mol)
        assert result.is_valid
        # Should have at least logP > 5
        assert result.logp > 5


class TestParseInvalid:
    """Test handling of invalid inputs."""

    def test_invalid_smiles(self):
        result = parse_molecule("not_a_molecule")
        assert result.is_valid is False
        assert result.canonical_smiles == ""
        assert result.molecular_weight is None

    def test_empty_string(self):
        result = parse_molecule("")
        assert result.is_valid is False

    def test_original_smiles_preserved(self):
        bad = "XYZ123"
        result = parse_molecule(bad)
        assert result.smiles == bad


class TestParseEdgeCases:
    """Test edge cases and special molecules."""

    def test_single_atom(self):
        result = parse_molecule("[Na]")
        assert result.is_valid
        assert result.num_rings == 0

    def test_charged_molecule(self):
        result = parse_molecule("[NH4+]")
        assert result.is_valid

    def test_to_dict(self):
        result = parse_molecule("CCO")
        d = result.to_dict()
        assert isinstance(d, dict)
        assert "canonical_smiles" in d
        assert "molecular_weight" in d
