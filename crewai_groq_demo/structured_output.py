import re

from pydantic import BaseModel

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def parse_structured_output[T: BaseModel](text: str, model: type[T]) -> T:
    """Parse an LLM's raw text response into `model`.

    Tasks are prompted to emit a JSON object directly instead of using
    CrewAI's `output_pydantic`/`InternalInstructor` path, which forces
    structured output via Groq's tool-calling protocol — every non-default
    model tried so far (qwen3-32b, llama-4-scout, gpt-oss-120b) broke on
    that path even though plain-text generation worked fine (see crew.py's
    `_kickoff_with_structured_output`). Parsing raw JSON ourselves is
    portable across any Groq model.

    Tolerates markdown code fences or stray prose around the JSON object,
    since models don't always follow "JSON only" instructions exactly.
    Raises `pydantic.ValidationError` (for both malformed JSON and schema
    mismatches) on failure, same as `model.model_validate_json` would.
    """
    candidate = text.strip()
    fence_match = _FENCE_RE.search(candidate)
    if fence_match:
        candidate = fence_match.group(1).strip()
    else:
        start, end = candidate.find("{"), candidate.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate = candidate[start : end + 1]
    return model.model_validate_json(candidate)
