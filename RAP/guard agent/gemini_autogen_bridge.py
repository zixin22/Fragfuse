"""
Route AutoGen AssistantAgent chat completions through Google GenAI (same relay as main.py).

pyautogen 0.2 wires ``generate_oai_reply`` to OpenAI Chat Completions. When defense uses Gemini,
we replace that reply handler so codegen uses only ``gemini_api.txt`` + GenAI ``base_url``.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple, Union

from autogen.agentchat.conversable_agent import ConversableAgent


def _function_tool_from_llm_config(llm_config: Any) -> Any:
    from google.genai import types

    funcs = []
    if isinstance(llm_config, dict):
        for spec in llm_config.get("functions") or []:
            name = spec.get("name")
            if not name:
                continue
            funcs.append(
                types.FunctionDeclaration(
                    name=name,
                    description=spec.get("description") or "",
                    parameters_json_schema=spec.get("parameters") or {"type": "object", "properties": {}},
                )
            )
    if not funcs:
        return None
    return types.Tool(function_declarations=funcs)


def _parse_openai_function_arguments(raw: Any) -> Dict[str, Any]:
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    s = str(raw).strip()
    if not s:
        return {}
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return {"cell": s}


def _oai_messages_to_gemini(
    messages: List[Dict[str, Any]],
) -> Tuple[Optional[str], List[Any]]:
    """Convert OpenAI-style chat messages to (system_instruction, genai contents list)."""
    from google.genai import types

    system_parts: List[str] = []
    contents: List[Any] = []
    for m in messages:
        role = m.get("role")
        if role == "system":
            c = m.get("content")
            if c:
                system_parts.append(str(c))
            continue
        if role == "user":
            text = m.get("content")
            if text is None:
                text = ""
            contents.append(types.Content(role="user", parts=[types.Part.from_text(text=str(text))]))
            continue
        if role == "assistant":
            fc = m.get("function_call")
            if fc:
                fd = dict(fc) if isinstance(fc, dict) else fc
                name = fd.get("name", "python")
                args = _parse_openai_function_arguments(fd.get("arguments"))
                contents.append(
                    types.Content(
                        role="model",
                        parts=[types.Part.from_function_call(name=name, args=args)],
                    )
                )
            else:
                text = m.get("content")
                if text is None:
                    text = ""
                contents.append(types.Content(role="model", parts=[types.Part.from_text(text=str(text))]))
            continue
        if role == "function":
            name = m.get("name") or "python"
            body = m.get("content")
            if body is None:
                body = ""
            part = types.Part.from_function_response(
                name=name,
                response={"result": str(body)},
            )
            contents.append(types.Content(role="tool", parts=[part]))
            continue
        text = m.get("content")
        if text:
            contents.append(types.Content(role="user", parts=[types.Part.from_text(text=str(text))]))

    system_instruction = "\n\n".join(system_parts) if system_parts else None
    return system_instruction, contents


def _gemini_reply_from_response(response: Any) -> Union[str, Dict[str, Any]]:
    fcs = getattr(response, "function_calls", None)
    if fcs:
        fc = fcs[0]
        name = getattr(fc, "name", None)
        args = getattr(fc, "args", None)
        if name is None and isinstance(fc, dict):
            name = fc.get("name")
            args = fc.get("args")
        if args is None:
            args = {}
        if not isinstance(args, dict):
            args = _parse_openai_function_arguments(args)
        return {
            "content": None,
            "function_call": {"name": name or "python", "arguments": json.dumps(args, ensure_ascii=False)},
        }
    text = getattr(response, "text", None)
    if text is not None and str(text).strip():
        return str(text).strip()
    return str(response).strip()


def gemini_generate_oai_reply(
    recipient: Any,
    messages: Optional[List[Dict[str, Any]]] = None,
    sender: Any = None,
    full_config: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, Union[str, Dict[str, Any], None]]:
    """Drop-in replacement for ``ConversableAgent.generate_oai_reply`` (Gemini only)."""
    if full_config is None:
        return False, None
    if messages is None:
        messages = recipient._oai_messages[sender]

    context = None
    if messages:
        last = messages[-1]
        if isinstance(last, dict) and "context" in last:
            context = last.pop("context", None)

    merged = list(recipient._oai_system_message) + list(messages)
    system_instruction, contents = _oai_messages_to_gemini(merged)

    gkey = full_config.get("gemini_api_key")
    if not gkey:
        raise ValueError("Gemini codegen path requires gemini_api_key in config.")
    gbase = full_config.get("gemini_base_url", "http://148.113.224.153:3000")
    gmodel = full_config.get("gemini_model") or "gemini-2.5-flash"

    os.environ["GEMINI_API_KEY"] = gkey
    try:
        from google import genai
        from google.genai import types
    except ImportError as e:
        raise ImportError("google-genai is required for Gemini AssistantAgent. pip install google-genai") from e

    client = genai.Client(http_options={"base_url": gbase})
    tool = _function_tool_from_llm_config(getattr(recipient, "llm_config", None))
    gen_cfg_kwargs: Dict[str, Any] = {
        "temperature": 0,
        "automatic_function_calling": types.AutomaticFunctionCallingConfig(disable=True),
    }
    if system_instruction:
        gen_cfg_kwargs["system_instruction"] = system_instruction
    if tool is not None:
        gen_cfg_kwargs["tools"] = [tool]

    config = types.GenerateContentConfig(**gen_cfg_kwargs)

    _ = context  # OpenAIWrapper uses context for templating; Gemini path does not today.

    response = client.models.generate_content(model=gmodel, contents=contents, config=config)
    reply = _gemini_reply_from_response(response)
    return True, reply


def patch_assistant_agent_for_gemini(agent: Any, full_config: Dict[str, Any]) -> None:
    """Swap AutoGen's OpenAI ``generate_oai_reply`` for :func:`gemini_generate_oai_reply`."""
    if not full_config.get("use_gemini_client"):
        return

    def _wrapped(recipient: Any, messages=None, sender=None, config=None):
        return gemini_generate_oai_reply(recipient, messages, sender, full_config)

    for item in agent._reply_func_list:
        if item["reply_func"] is ConversableAgent.generate_oai_reply:
            item["reply_func"] = _wrapped
            return
    raise RuntimeError("patch_assistant_agent_for_gemini: generate_oai_reply not found on agent")
