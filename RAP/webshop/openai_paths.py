"""Canonical webshop paths for OpenAI-compatible API keys (file only; no environment variables)."""

import os

WEBSHOP_DIR = os.path.abspath(os.path.dirname(__file__))
OPENAI_API_KEY_PATH = os.path.join(WEBSHOP_DIR, "OpenAI_api_key.txt")


def read_openai_api_key() -> str:
    """Read API key from ``webshop/OpenAI_api_key.txt`` only."""
    if not os.path.isfile(OPENAI_API_KEY_PATH):
        raise FileNotFoundError(
            f"OpenAI API key file not found: {OPENAI_API_KEY_PATH}"
        )
    with open(OPENAI_API_KEY_PATH, "rb") as f:
        raw = f.read()
    if raw.startswith(b"\xef\xbb\xbf"):
        raw = raw[3:]
    key = raw.decode("utf-8", errors="strict").strip()
    if key.startswith("\ufeff"):
        key = key.lstrip("\ufeff")
    if not key:
        raise ValueError(f"OpenAI API key file is empty: {OPENAI_API_KEY_PATH}")
    try:
        key.encode("ascii")
    except UnicodeEncodeError as exc:
        raise ValueError(
            "API key contains non-ASCII characters; remove BOM/invisible chars."
        ) from exc
    return key
