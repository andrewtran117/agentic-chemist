"""Tests for compute_similarity tool."""

import pytest
from agentic_chemist.tools.similarity import compute_similarity


class TestSimilarityValid:
    """Test similarity scores for known molecule pairs."""

    def test_identical_molecules(self):
        aspirin = "CC(=O)Oc1ccccc1C(=O)O"
        assert compute_similarity(aspirin, aspirin) == 1.0

    def test_same_molecule_different_smiles(self):
        benzene_a = "c1ccccc1"
        benzene_b = "C1=CC=CC=C1"
        assert compute_similarity(benzene_a, benzene_b) == 1.0

    def test_similar_molecules(self):
        # Benzene vs toluene — structurally close
        benzene = "c1ccccc1"
        toluene = "Cc1ccccc1"
        sim = compute_similarity(benzene, toluene)
        assert 0.2 < sim < 1.0  # similar but not identical; small molecules have fewer fingerprint bits

    def test_dissimilar_molecules(self):
        # Ethanol vs a large steroid (cholesterol)
        ethanol = "CCO"
        cholesterol = "CC(C)CCCC(C)C1CCC2C1(CCC3C2CC=C4C3(CCC(C4)O)C)C"
        sim = compute_similarity(ethanol, cholesterol)
        assert sim < 0.2  # very different

    def test_symmetry(self):
        a = "CC(=O)Oc1ccccc1C(=O)O"  # aspirin
        b = "CCO"  # ethanol
        assert compute_similarity(a, b) == compute_similarity(b, a)

    def test_result_in_range(self):
        sim = compute_similarity("CCO", "CCCO")
        assert 0.0 <= sim <= 1.0


class TestSimilarityInvalid:
    """Test handling of invalid inputs."""

    def test_invalid_first(self):
        assert compute_similarity("not_valid", "CCO") is None

    def test_invalid_second(self):
        assert compute_similarity("CCO", "not_valid") is None

    def test_both_invalid(self):
        assert compute_similarity("bad", "worse") is None
