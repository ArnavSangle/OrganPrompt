import os
import anthropic

_client = None

def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    return _client

_SYSTEM_PROMPT = """You are a medical imaging expert writing captions for CT scan images.
Given a base anatomical description, produce exactly N rephrased variants.
Rules:
- Preserve all factual anatomical content (organ name, location, imaging appearance)
- Vary sentence structure, word order, and phrasing only
- Keep each variant under 100 words
- Output ONLY the variants, one per line, no numbering, no preamble"""

def generate_caption_variants(template: str, n: int = 3) -> list[str]:
    """Call Claude to produce N rephrased variants of the given template caption.

    Returns a list of exactly n strings. Falls back to [template] * n if the
    API returns fewer lines than requested.
    """
    client = _get_client()
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=512,
        system=_SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": f"Produce {n} variants of this caption:\n\n{template}"
            }
        ]
    )
    raw = response.content[0].text.strip()
    lines = [line.strip().lstrip("0123456789.-) ") for line in raw.split("\n") if line.strip()]
    while len(lines) < n:
        lines.append(template)
    return lines[:n]
