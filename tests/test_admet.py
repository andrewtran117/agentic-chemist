"""Tests for predict_admet tool."""

import pytest
from agentic_chemist.tools.admet import predict_admet


class TestSolubility:
    """Test solubility predictions."""

    def test_aspirin_solubility(self):
        result = predict_admet("CC(=O)Oc1ccccc1C(=O)O", "solubility")
        assert result is not None
        assert result.endpoint == "solubility"
        assert result.units == "logS"
        # Aspirin is moderately soluble; logS roughly -1 to -3
        assert -5 < result.value < 1

    def test_caffeine_solubility(self):
        result = predict_admet("Cn1c(=O)c2c(ncn2C)n(C)c1=O", "solubility")
        assert result is not None
        # Caffeine is fairly soluble
        assert result.value > -4

    def test_solubility_has_applicability(self):
        result = predict_admet("CCO", "solubility")
        assert result.applicability_flag in ("in_domain", "borderline", "out_of_domain")
        assert 0.0 <= result.nearest_train_tanimoto <= 1.0


class TestHERG:
    """Test hERG predictions."""

    def test_herg_returns_probability(self):
        result = predict_admet("CC(=O)Oc1ccccc1C(=O)O", "herg")
        assert result is not None
        assert result.endpoint == "herg"
        assert result.units == "probability"
        assert 0.0 <= result.value <= 1.0

    def test_herg_model_id(self):
        result = predict_admet("CCO", "herg")
        assert "herg" in result.model_id


class TestApplicability:
    """Test applicability domain flagging."""

    def test_common_drug_in_domain(self):
        # Aspirin is a common druglike molecule — should be in domain
        result = predict_admet("CC(=O)Oc1ccccc1C(=O)O", "solubility")
        assert result.applicability_flag == "in_domain"

    def test_exotic_molecule_flagged(self):
        # A weird ferrocene-like organometallic — unlikely in AqSolDB training set
        result = predict_admet("[Fe+2].[cH-]1cccc1.[cH-]1cccc1", "solubility")
        if result is not None:
            # Should be borderline or out-of-domain
            assert result.nearest_train_tanimoto < 0.5


class TestInvalidInputs:
    """Test error handling."""

    def test_invalid_smiles(self):
        result = predict_admet("not_a_molecule", "solubility")
        assert result is None

    def test_unknown_endpoint(self):
        with pytest.raises(ValueError, match="Unknown endpoint"):
            predict_admet("CCO", "bogus_endpoint")

    def test_to_dict(self):
        result = predict_admet("CCO", "solubility")
        d = result.to_dict()
        assert isinstance(d, dict)
        assert "applicability_flag" in d
        assert "nearest_train_tanimoto" in d
