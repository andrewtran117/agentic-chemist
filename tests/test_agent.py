"""Tests for the agent loop components."""

import json
import pytest

from agentic_chemist.tools.registry import get_tool_schemas, call_tool, TOOL_REGISTRY
from agentic_chemist.agent.trace import Trace


class TestRegistry:
    """Test tool registry."""

    def test_all_tools_registered(self):
        assert "parse_molecule" in TOOL_REGISTRY
        assert "compute_similarity" in TOOL_REGISTRY
        assert "predict_admet" in TOOL_REGISTRY

    def test_schemas_valid_for_api(self):
        schemas = get_tool_schemas()
        assert len(schemas) == 3
        for schema in schemas:
            assert "name" in schema
            assert "description" in schema
            assert "input_schema" in schema
            assert schema["input_schema"]["type"] == "object"
            assert "properties" in schema["input_schema"]
            assert "required" in schema["input_schema"]

    def test_call_tool_parse(self):
        result = call_tool("parse_molecule", {"smiles": "CCO"})
        assert result.is_valid
        assert result.molecular_weight > 0

    def test_call_tool_similarity(self):
        result = call_tool("compute_similarity", {"smiles_a": "CCO", "smiles_b": "CCCO"})
        assert 0 < result < 1

    def test_call_tool_admet(self):
        result = call_tool("predict_admet", {"smiles": "CCO", "endpoint": "solubility"})
        assert result is not None
        assert result.endpoint == "solubility"

    def test_call_unknown_tool(self):
        with pytest.raises(ValueError, match="Unknown tool"):
            call_tool("nonexistent", {})

    def test_predict_admet_schema_has_enum(self):
        schema = TOOL_REGISTRY["predict_admet"]["schema"]
        endpoint_prop = schema["input_schema"]["properties"]["endpoint"]
        assert "enum" in endpoint_prop
        assert "solubility" in endpoint_prop["enum"]
        assert "herg" in endpoint_prop["enum"]


class TestTrace:
    """Test trace logging."""

    def test_trace_records_tool_calls(self):
        trace = Trace(task="test task")
        trace.add_tool_call("parse_molecule", {"smiles": "CCO"}, {"valid": True}, 5.0)
        assert trace.step_count == 1
        assert trace.steps[0].tool_name == "parse_molecule"
        assert trace.steps[0].step == 1

    def test_trace_accumulates_tokens(self):
        trace = Trace(task="test")
        trace.add_usage(100, 50)
        trace.add_usage(200, 75)
        assert trace.total_input_tokens == 300
        assert trace.total_output_tokens == 125

    def test_trace_finalize(self):
        trace = Trace(task="test")
        trace.finalize("Final answer")
        assert trace.final_response == "Final answer"
        assert trace.end_time is not None
        assert trace.total_latency_ms >= 0

    def test_trace_serialization(self):
        trace = Trace(task="test")
        trace.add_tool_call("parse_molecule", {"smiles": "CCO"}, {"valid": True}, 5.0)
        trace.finalize("done")

        d = trace.to_dict()
        assert d["task"] == "test"
        assert d["step_count"] == 1
        assert len(d["steps"]) == 1

        j = trace.to_json()
        parsed = json.loads(j)
        assert parsed["task"] == "test"

    def test_trace_converts_dataclass_output(self):
        from agentic_chemist.tools.parse import parse_molecule
        result = parse_molecule("CCO")
        trace = Trace(task="test")
        trace.add_tool_call("parse_molecule", {"smiles": "CCO"}, result, 1.0)
        # Should have been converted to dict
        assert isinstance(trace.steps[0].output, dict)
        assert "canonical_smiles" in trace.steps[0].output

    def test_trace_records_errors(self):
        trace = Trace(task="test")
        trace.add_tool_call("bad_tool", {}, None, 1.0, error="Tool not found")
        assert trace.steps[0].error == "Tool not found"

    def test_multiple_steps_numbered(self):
        trace = Trace(task="test")
        trace.add_tool_call("a", {}, {}, 1.0)
        trace.add_tool_call("b", {}, {}, 1.0)
        trace.add_tool_call("c", {}, {}, 1.0)
        assert [s.step for s in trace.steps] == [1, 2, 3]
