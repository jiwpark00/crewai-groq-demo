import pytest
from pydantic import BaseModel, ValidationError

from crewai_groq_demo.structured_output import parse_structured_output


class _Sample(BaseModel):
    name: str
    count: int


def test_parses_clean_json() -> None:
    result = parse_structured_output('{"name": "a", "count": 1}', _Sample)

    assert result == _Sample(name="a", count=1)


def test_strips_markdown_json_fence() -> None:
    text = '```json\n{"name": "a", "count": 1}\n```'

    result = parse_structured_output(text, _Sample)

    assert result == _Sample(name="a", count=1)


def test_strips_plain_markdown_fence() -> None:
    text = '```\n{"name": "a", "count": 1}\n```'

    result = parse_structured_output(text, _Sample)

    assert result == _Sample(name="a", count=1)


def test_strips_surrounding_prose_without_fence() -> None:
    text = 'Here is the JSON:\n{"name": "a", "count": 1}\nLet me know if you need more.'

    result = parse_structured_output(text, _Sample)

    assert result == _Sample(name="a", count=1)


def test_raises_validation_error_on_malformed_json() -> None:
    with pytest.raises(ValidationError):
        parse_structured_output("not json at all", _Sample)


def test_raises_validation_error_on_schema_mismatch() -> None:
    with pytest.raises(ValidationError):
        parse_structured_output('{"name": "a"}', _Sample)
