# GuardAgent WebShop Defense

This folder contains the GuardAgent implementation used by the WebShop defense path.

The main experiment entrypoint is still:

```bash
cd ../webshop
python main.py --model gpt-4o --attack --dataset dataset_test_12.json --defense_mode guard_agent --defense_check_target instruction --retrieve_mode rap --limit 100 --output output/guardagent_29
```

`webshop/main.py` imports `WebShopGuardAgent` from this folder when `--defense_mode guard_agent` is enabled.

## Files

- `webshop_guard_agent.py`: WebShop-facing adapter with the same `check_all_rules(...)` surface as `RuleChecker`.
- `guardagent.py`: AutoGen `UserProxyAgent` subclass for task decomposition, code generation handoff, and tool execution.
- `toolset_webshop.py`: Executes generated WebShop guardrail code and normalizes results.
- `tools.py`: Guardrail helper functions exposed to generated code.
- `prompts_guard.py`: Shared GuardAgent prompts and generated-code header.
- `config.py`: LLM and API-key configuration for OpenAI-compatible and Gemini paths.
- `gemini_autogen_bridge.py`: Gemini adapter for AutoGen code generation.

API keys are read from `../webshop/OpenAI_api_key.txt` and `../webshop/gemini_api.txt`.
