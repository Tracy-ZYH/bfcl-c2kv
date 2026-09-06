import json

from c2kv_eval.adapters.sglang_tool_schema import normalize_tools_for_sglang


def test_normalization_matches_sglang_pydantic_dump_shape() -> None:
    source = [
        {
            "ignored": "outer",
            "function": {
                "name": "lookup",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                },
                "response": {"type": "string"},
                "description": "Look up a value.",
                "ignored": "inner",
            },
            "type": "function",
        }
    ]

    normalized = normalize_tools_for_sglang(source)

    assert normalized == [
        {
            "type": "function",
            "function": {
                "description": "Look up a value.",
                "name": "lookup",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                },
                "strict": False,
            },
        }
    ]
    assert list(normalized[0]) == ["type", "function"]
    assert list(normalized[0]["function"]) == [
        "description",
        "name",
        "parameters",
        "strict",
    ]
    assert source[0]["function"]["response"] == {"type": "string"}


def test_preserves_explicit_strict_and_optional_defaults() -> None:
    source = [
        {
            "type": "function",
            "function": {
                "strict": True,
                "parameters": None,
                "name": "strict_lookup",
            },
        }
    ]

    normalized = normalize_tools_for_sglang(source)

    assert normalized[0]["function"]["description"] is None
    assert normalized[0]["function"]["strict"] is True
    assert json.dumps(normalized, separators=(",", ":")) == (
        '[{"type":"function","function":{"description":null,'
        '"name":"strict_lookup","parameters":null,"strict":true}}]'
    )
