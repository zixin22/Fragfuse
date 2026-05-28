"""GuardAgent LLM config: OpenAI-compatible path or pure Gemini (GenAI for all stages when model name matches)."""

import os

# Same relays as webshop/main.py and rule_checker.py
OPENAI_COMPAT_BASE_URL = "http://152.53.53.64:3000/v1"
GEMINI_GENAI_BASE_URL = "http://148.113.224.153:3000"


def _webshop_dir() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "webshop"))


def _read_openai_key_file() -> str:
    path = os.path.join(_webshop_dir(), "OpenAI_api_key.txt")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"OpenAI API key file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        key = f.read().strip()
    if not key:
        raise ValueError(f"OpenAI API key file is empty: {path}")
    return key


def _read_gemini_key_file() -> str:
    path = os.path.join(_webshop_dir(), "gemini_api.txt")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Gemini API key file not found: {path}")
    with open(path, "rb") as f:
        raw = f.read()
    if raw.startswith(b"\xef\xbb\xbf"):
        raw = raw[3:]
    key = raw.decode("utf-8", errors="strict").strip()
    if key.startswith("\ufeff"):
        key = key.lstrip("\ufeff")
    if not key:
        raise ValueError(f"Gemini API key file is empty: {path}")
    return key


def model_config(model: str):
    """
    Build config_list[0] for GuardAgent + AutoGen.

    - OpenAI-style models: OPENAI_COMPAT_BASE_URL + ``webshop/OpenAI_api_key.txt`` only.
    - Gemini (*gemini* in name): GEMINI_GENAI_BASE_URL + ``webshop/gemini_api.txt`` only
      (decomposition, debugger, and AutoGen codegen via ``gemini_autogen_bridge``).
    """
    ml = (model or "").lower()

    if "gemini" in ml:
        gemini_key = _read_gemini_key_file()
        gemini_model = model.strip() or "gemini-2.5-flash"
        return {
            "use_gemini_client": True,
            "gemini_api_key": gemini_key,
            "gemini_base_url": GEMINI_GENAI_BASE_URL,
            "gemini_model": gemini_model,
        }

    openai_key = _read_openai_key_file()

    if "gpt-3.5-turbo" in ml:
        model_name = "gpt-3.5-turbo"
    elif "gpt-4o" in ml:
        model_name = "gpt-4o"
    elif "gpt-4" in ml:
        model_name = "gpt-4"
    else:
        model_name = "gpt-4"

    return {
        "model": model_name,
        "api_key": openai_key,
        "base_url": OPENAI_COMPAT_BASE_URL,
        "use_gemini_client": False,
    }


_AUTOGEN_OPENAI_KEYS = frozenset({"model", "api_key", "base_url", "api_type", "api_version"})


def openai_config_for_autogen(full: dict) -> dict:
    """
    Subset of ``model_config`` for AutoGen's OpenAIWrapper.

    Gemini defense still constructs an OpenAIWrapper (for ``functions`` / llm_config shape) but
    ``generate_oai_reply`` is patched to GenAI before any request; placeholders avoid reading
    OpenAI keys.
    """
    if full.get("use_gemini_client"):
        return {
            "model": full.get("gemini_model", "gemini-2.5-flash"),
            "api_key": "unused-gemini-codegen-patched",
            "base_url": "http://127.0.0.1:9/v1",
        }
    return {k: full[k] for k in _AUTOGEN_OPENAI_KEYS if k in full}


def llm_config_list(seed, config_list):
    llm_config_list = {
        "functions": [
            {
                "name": "python",
                "description": "run the entire code and return the execution result. Only generate the code.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "cell": {
                            "type": "string",
                            "description": "Valid Python code to execute.",
                        }
                    },
                    "required": ["cell"],
                },
            },
        ],
        "config_list": config_list,
        "timeout": 120,
        "cache_seed": seed,
        "temperature": 0,
    }
    return llm_config_list
