"""Tool-schema compatibility must preserve accepted inputs and public contracts."""

from copy import deepcopy

from jsonschema import Draft202012Validator

from src.agent.llm_adapter import _normalize_nullable_string_enums
from src.agent.tools.fxmacrodata_tools import build_fxmacrodata_tools


def test_macro_enum_schemas_preserve_valid_and_invalid_arguments():
    for definition in build_fxmacrodata_tools():
        original = definition.to_openai_tool()
        snapshot = deepcopy(original)
        normalized = _normalize_nullable_string_enums([original])[0]
        before = original["function"]["parameters"]
        after = normalized["function"]["parameters"]
        for name, schema in before.get("properties", {}).items():
            if "anyOf" not in schema or "enum" not in schema:
                continue
            validators = [Draft202012Validator(schema), Draft202012Validator(after["properties"][name])]
            for value in schema["enum"] + [None, "invalid-value", 1, True, [], {}]:
                assert validators[0].is_valid(value) == validators[1].is_valid(value)
        assert original == snapshot
        assert definition.to_openai_tool() == snapshot
        assert definition.to_mcp_descriptor()["inputSchema"] == before


def test_nested_schema_repairs_keep_nullable_values_and_branch_constraints():
    enum_schema = {"anyOf": [{"type": "string"}, {"type": "null"}], "enum": ["monthly"]}
    nullable = {"anyOf": [{"type": "string"}, {"type": "null"}]}
    constrained = {"anyOf": [{"type": "string", "minLength": 3}, {"type": "null"}], "enum": ["a"]}
    schema = {
        "type": "object",
        "properties": {
            "filters": {"type": "array", "items": {"type": "object", "properties": {"period": enum_schema}}},
            "nullable": nullable,
            "constrained": constrained,
        },
        "default": {"anyOf": [{"type": "string"}, {"type": "null"}], "enum": ["literal-data"]},
    }
    tool = {"type": "function", "function": {"name": "nested", "parameters": schema}}
    result = _normalize_nullable_string_enums([tool])[0]["function"]["parameters"]
    for value in [{}, {"filters": [{"period": "monthly"}]}, {"filters": [{"period": None}]},
                  {"nullable": None}, {"constrained": "a"}]:
        assert Draft202012Validator(schema).is_valid(value) == Draft202012Validator(result).is_valid(value)
    assert result["properties"]["filters"]["items"]["properties"]["period"]["type"] == "string"
    assert result["properties"]["nullable"] == nullable
    assert result["properties"]["constrained"] == constrained
    assert result["default"] == schema["default"]
