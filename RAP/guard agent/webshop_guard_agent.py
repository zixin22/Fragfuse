"""
WebShop GuardAgent adapter (“WebShop Guard”)

**WebShop Guard** is this stack: AutoGen **GuardAgent** (task decomposition + error diagnosis)
plus an **AssistantAgent** that proposes guardrail Python (invoking ``run_code_webshop``), wired for
WebShop rule checks. ``main.py`` enables it with ``--defense_mode guard_agent``; it exposes the
same ``check_all_rules(...)`` surface as **RuleChecker** so defense hooks stay interchangeable.

When ``--defense_mode_model`` names a **Gemini** model, decomposition, debugger, and AutoGen
codegen all use **Google GenAI** only (``gemini_api.txt``, same relay as ``main.py``).
"""

import os
import sys
import json
import re
import ast
from typing import Dict, List, Tuple, Optional, Union
import autogen

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
WEBSHOP_DIR = os.path.join(PROJECT_ROOT, "webshop")
RULE_PROFILE_DIR = os.path.join(WEBSHOP_DIR, "rule_and_profile")

for path in (CURRENT_DIR, WEBSHOP_DIR, RULE_PROFILE_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

from guardagent import GuardAgent
from config import model_config, llm_config_list, openai_config_for_autogen
from gemini_autogen_bridge import patch_assistant_agent_for_gemini
from toolset_webshop import run_code_webshop
from user_profile import UserProfile


class WebShopGuardAgent:
    """
    WebShop-facing wrapper around AutoGen GuardAgent + codegen assistant (“WebShop Guard”).

    Implements ``check_all_rules`` like ``RuleChecker`` so
    ``main.py`` can swap defense backends without changing the WebShop loop.
    """
    
    def __init__(self, verbose: bool = False, model: str = "gpt-4", num_shots: int = 3, seed: int = 42):
        """
        Initialize WebShop GuardAgent
        
        Args:
            verbose: Whether to print verbose output
            model: LLM for Guard — OpenAI-style names use ``OpenAI_api_key.txt``; names containing
                ``gemini`` use ``gemini_api.txt`` for all Guard stages including codegen.
            num_shots: Number of few-shot examples to use (1, 2, or 3)
            seed: Random seed for reproducibility
        """
        self.verbose = verbose
        self.model = model
        self.num_shots = num_shots
        self.seed = seed
        
        # Full config for GuardAgent (task decomposition / debugger); slim copy for AutoGen chatbot.
        _full = model_config(model)
        config_list = [_full]
        llm_config = llm_config_list(seed, [openai_config_for_autogen(_full)])
        
        # Create chatbot agent (code generator)
        self.chatbot = autogen.agentchat.AssistantAgent(
            name="chatbot",
            system_message=(
                "For coding tasks, only use the functions you have been provided with. "
                "Prefer calling the `python` tool with JSON {\"cell\": \"...\"} so guardrail code runs with the correct imports. "
                "Reply TERMINATE when the task is done."
            ),
            llm_config=llm_config,
        )
        if _full.get("use_gemini_client"):
            patch_assistant_agent_for_gemini(self.chatbot, _full)
        
        # Create GuardAgent instance
        # code_execution_config=False: AutoGen's markdown code-block runner does NOT prepend WebShop
        # CodeHeader (CheckRule / tools), which caused NameError when the model returned ```python ... ```
        # instead of a python tool call. Execution must go through register_function -> run_code_webshop.
        self.guard_agent = GuardAgent(
            name="user_proxy",
            is_termination_msg=lambda x: x.get("content", "") and x.get("content", "").rstrip().endswith("TERMINATE"),
            human_input_mode="NEVER",
            max_consecutive_auto_reply=3,
            code_execution_config=False,
            config_list=config_list,
        )
        
        # Register WebShop-specific function
        self.guard_agent.register_function(
            function_map={
                "python": run_code_webshop
            }
        )
        
        # Initialize long-term memory
        self._init_memory()
        
        # Load WebShop-specific prompts
        self._load_webshop_prompts()
    
    def _init_memory(self):
        """Initialize long-term memory with WebShop examples"""
        try:
            from request_webshop import CodeGEN_Examples
            init_memory = CodeGEN_Examples
        except ImportError:
            try:
                from rule_and_profile.request_webshop import CodeGEN_Examples
                init_memory = CodeGEN_Examples
            except ImportError:
                # If no examples file, start with empty memory
                init_memory = ""
        
        self.long_term_memory = []
        if init_memory:
            # Split by full example boundary instead of blank lines.
            # CodeGEN_Examples contains many blank lines inside one example.
            blocks = re.split(r'\n(?=Agent input:\n)', init_memory.strip())
            for block in blocks:
                block = block.strip()
                if not block:
                    continue
                try:
                    if not block.startswith("Agent input:\n"):
                        block = "Agent input:\n" + block
                    item = block.split('Agent input:\n', 1)[1]
                    agent_input, rest = item.split('\nAgent output:\n', 1)
                    agent_output, rest = rest.split('\nTask decomposition:\n', 1)
                    subtasks, code = rest.split('\nGuardrail code:\n', 1)
                    new_item = {
                        "agent input": agent_input.strip(),
                        "agent output": agent_output.strip(),
                        "subtasks": subtasks.strip(),
                        "code": code.strip()
                    }
                    self.long_term_memory.append(new_item)
                except Exception as e:
                    if self.verbose:
                        print(f"[Warning] Failed to parse memory item: {e}")
                    continue
    
    def _load_webshop_prompts(self):
        """Load WebShop-specific prompts"""
        try:
            from request_webshop import (
                User_Request_WebShop,
                Specification_WebShop,
                Decomposition_Examples
            )
            self.user_request = User_Request_WebShop
            self.agent_specification = Specification_WebShop
            self.decomposition_examples = Decomposition_Examples
        except ImportError:
            try:
                from rule_and_profile.request_webshop import (
                    User_Request_WebShop,
                    Specification_WebShop,
                    Decomposition_Examples
                )
                self.user_request = User_Request_WebShop
                self.agent_specification = Specification_WebShop
                self.decomposition_examples = Decomposition_Examples
            except ImportError:
                # Fallback to default prompts if file doesn't exist
                self.user_request = "Check if the user purchase request violates business rules."
                self.agent_specification = "WebShop is an e-commerce platform."
                self.decomposition_examples = ""

    def check_all_rules(self, profile: UserProfile, instruction: str, query: str,
                       return_details: bool = False) -> Union[Tuple[bool, List[str]], Tuple[bool, List[str], Dict]]:
        """
        Check all rules against the purchase request.
        This method provides the same interface as RuleChecker.check_all_rules()
        
        Args:
            profile: UserProfile object containing user attributes
            instruction: The instruction text describing what the user wants to buy
            query: The query/category text (e.g., "fresh meal kits", "headphones")
            return_details: If True, return additional details (prompt and response)
        
        Returns:
            If return_details=False: Tuple of (is_valid, list_of_violated_rules)
            If return_details=True: Tuple of (is_valid, list_of_violated_rules, details_dict)
            - is_valid: True if no rules violated, False otherwise
            - list_of_violated_rules: List of violated rule names
            - details_dict: Dictionary with 'prompt' and 'response' (if return_details=True)
        """
        try:
            # Prepare agent input and output
            agent_input = self._format_agent_input(profile, instruction, query)
            agent_output = self._format_agent_output(instruction, query)
            
            # Update guard agent memory
            self.guard_agent.update_memory(self.num_shots, self.long_term_memory)
            
            # CRITICAL: Initialize _oai_messages dictionaries bidirectionally BEFORE clearing
            # chatbot._oai_messages needs guard_agent as key (for clear_history call in _prepare_chat)
            if not hasattr(self.chatbot, '_oai_messages') or self.chatbot._oai_messages is None:
                self.chatbot._oai_messages = {}
            if not hasattr(self.guard_agent, '_oai_messages') or self.guard_agent._oai_messages is None:
                self.guard_agent._oai_messages = {}
            
            # Initialize bidirectional entries BEFORE clearing (clear_history needs these keys)
            if self.guard_agent not in self.chatbot._oai_messages:
                self.chatbot._oai_messages[self.guard_agent] = []
            if self.chatbot not in self.guard_agent._oai_messages:
                self.guard_agent._oai_messages[self.chatbot] = []
            
            # Clear message lists (but keep dictionary structure with keys)
            self.guard_agent._oai_messages[self.chatbot].clear()
            self.chatbot._oai_messages[self.guard_agent].clear()
            
            # Ensure chat_messages and reply_at_receive dictionaries are initialized
            # These dictionaries use recipient (chatbot) as key, so we need to initialize them
            if not hasattr(self.guard_agent, 'chat_messages'):
                self.guard_agent.chat_messages = {}
            if not hasattr(self.guard_agent, 'reply_at_receive'):
                self.guard_agent.reply_at_receive = {}
            
            # Initialize entries for chatbot if they don't exist
            if self.chatbot not in self.guard_agent.chat_messages:
                self.guard_agent.chat_messages[self.chatbot] = []
            if self.chatbot not in self.guard_agent.reply_at_receive:
                self.guard_agent.reply_at_receive[self.chatbot] = True
            
            # Initiate chat with GuardAgent
            # Verify all required attributes are set
            if not hasattr(self, 'user_request') or self.user_request is None:
                raise ValueError("user_request is not initialized. Call _load_webshop_prompts() first.")
            if not hasattr(self, 'agent_specification') or self.agent_specification is None:
                raise ValueError("agent_specification is not initialized. Call _load_webshop_prompts() first.")
            if not hasattr(self, 'decomposition_examples') or self.decomposition_examples is None:
                self.decomposition_examples = ""
            
            try:
                self.guard_agent.initiate_chat(
                    self.chatbot,
                    user_request=self.user_request,
                    agent_specification=self.agent_specification,
                    agent_input=agent_input,
                    agent_output=agent_output,
                    agent_task_deco_examples=self.decomposition_examples,
                )
            except Exception as chat_error:
                # Handle initiate_chat errors separately
                error_type = type(chat_error).__name__
                error_details = str(chat_error)
                
                if self.verbose:
                    print(f"[Error] GuardAgent initiate_chat failed: {error_type}: {error_details}")
                    import traceback
                    traceback.print_exc()
                
                # Extract error message safely
                try:
                    error_msg = error_details if error_details else 'Unknown error'
                    # For KeyError, check first (before cleaning object representations)
                    if isinstance(chat_error, KeyError):
                        # KeyError can have the key in args[0] or as a string representation
                        if chat_error.args:
                            error_arg = chat_error.args[0]
                            # Check if args[0] is an object (like AssistantAgent) instead of a string
                            if not isinstance(error_arg, str):
                                # This is likely a KeyError from accessing a dict with an object key
                                error_msg = f"KeyError: Missing key in dictionary (key is an object: {type(error_arg).__name__}). This usually means chat_messages or reply_at_receive dictionaries are not properly initialized for the recipient (chatbot)."
                            # Check if args[0] is a string containing "Missing required keys"
                            elif "Missing required keys" in error_arg:
                                # Extract the list of missing keys from the error message
                                import re
                                match = re.search(r'\[(.*?)\]', error_arg)
                                if match:
                                    missing_keys_str = match.group(1)
                                    missing_keys = [k.strip().strip("'\"") for k in missing_keys_str.split(',')]
                                    error_msg = f"KeyError: Missing required keys in GuardAgent initiate_chat context: {missing_keys}"
                                else:
                                    error_msg = f"KeyError: {error_arg}"
                            else:
                                # args[0] is the missing key itself (string)
                                missing_key = error_arg
                                error_msg = f"KeyError: Missing required key '{missing_key}' in GuardAgent initiate_chat context"
                        else:
                            # Try to extract from string representation
                            error_str = str(chat_error)
                            error_msg = f"KeyError: {error_str}"
                    # Handle AuthenticationError specifically (API key issues)
                    elif 'AuthenticationError' in error_type or '401' in error_details or 'invalid_api_key' in error_details:
                        error_msg = (
                            f"AuthenticationError: Invalid or expired API key. "
                            f"For OpenAI models use webshop/OpenAI_api_key.txt; for Gemini-only defense use webshop/gemini_api.txt. "
                            f"Error: {error_details[:200]}"
                        )
                    # Clean up error message if it contains object representations (for non-KeyError)
                    elif '<' in error_msg and 'object at 0x' in error_msg:
                        error_msg = f"{error_type}: GuardAgent initiate_chat failed"
                    else:
                        error_msg = f"{error_type}: {error_details}"
                except Exception as parse_error:
                    if self.verbose:
                        print(f"[Debug] Failed to parse error: {parse_error}")
                        import traceback
                        traceback.print_exc()
                    error_msg = f"{error_type}: GuardAgent initiate_chat failed (failed to extract details: {parse_error})"
                raise Exception(f"GuardAgent initiate_chat error: {error_msg}") from chat_error
            
            # Extract results from GuardAgent response (both agents; see _collect_guardagent_log_chunks)
            logs_string = self._collect_guardagent_log_chunks()
            logs_string = self._maybe_salvage_markdown_python(logs_string)
            
            # Parse GuardAgent results
            is_valid, violated_rules = self._parse_guard_agent_results(logs_string)

            # Update memory if check was successful
            if is_valid:
                self._update_memory(agent_input, agent_output, logs_string)
            
            if self.verbose:
                if violated_rules:
                    print(f"[GuardAgent] Rules violated: {', '.join(violated_rules)}")
                else:
                    print(f"[GuardAgent] No rules violated")
            
            if return_details:
                # Build details dict similar to RuleChecker
                # Extract response from logs_string if available
                response_text = 'VALID'
                if violated_rules:
                    response_text = f"VIOLATED:{','.join(violated_rules)}"
                else:
                    # Try to extract response from logs
                    for log_item in logs_string:
                        if isinstance(log_item, str) and "GuardAgent results:" in log_item:
                            if "action_denied: 1" in log_item or "action_denied: True" in log_item:
                                response_text = f"VIOLATED:{','.join(violated_rules)}"
                            break
                
                # Ensure response_text is always a string (safety check)
                if not isinstance(response_text, str):
                    response_text = str(response_text)
                    # Clean up if it contains object representations
                    if '<' in response_text and 'object at 0x' in response_text:
                        response_text = 'VALID' if not violated_rules else f"VIOLATED:{','.join(violated_rules)}"
                
                details = {
                    'prompt': f"User Profile:\n{agent_input}\n\nPurchase Request:\n{agent_output}",
                    'response': response_text
                }
                return is_valid, violated_rules, details
            else:
                return is_valid, violated_rules
            
        except Exception as e:
            error_type = type(e).__name__
            error_details = str(e)
            
            if self.verbose:
                print(f"[Error] GuardAgent check failed: {error_type}: {error_details}")
                import traceback
                traceback.print_exc()
            
            # Security-first approach: On error, default to invalid (block session)
            # This prevents attacks when GuardAgent fails
            # Extract error message safely (handle case where exception might contain objects)
            try:
                error_msg = error_details if error_details else 'Unknown error'
                # If error message contains object representation, extract just the error type
                if '<' in error_msg and 'object at 0x' in error_msg:
                    error_msg = f"{error_type}: GuardAgent execution failed"
                # For KeyError, include the missing key
                elif isinstance(e, KeyError):
                    missing_key = e.args[0] if e.args else str(e)
                    error_msg = f"KeyError: Missing required key '{missing_key}' in GuardAgent execution"
                else:
                    error_msg = f"{error_type}: {error_details}"
            except Exception as parse_error:
                error_msg = f"{error_type}: GuardAgent execution failed (failed to extract details: {parse_error})"
            
            if return_details:
                return False, ['guardagent_exception'], {'prompt': '', 'response': f'Error: {error_msg}'}
            else:
                return False, ['guardagent_exception']
    
    def _format_agent_input(self, profile: UserProfile, instruction: str, query: str) -> str:
        """Format agent input for GuardAgent"""
        profile_dict = profile.to_dict()
        return f"""User Profile:
- Profile ID: {profile_dict['profile_id']}
- Age: {profile_dict['age']}
- Country: {profile_dict['country']}
- Is Verified: {profile_dict['is_verified']}
- Payment Method: {profile_dict['payment_method']}
- Failed Payment Attempts: {profile_dict['failed_payment_attempts']}
- Credit Score: {profile_dict['credit_score']}
- Account Age (days): {profile_dict['account_age_days']}
- Account Status: {profile_dict['account_status']}
- Return Rate (%): {profile_dict['return_rate']:.1f}
- Total Purchase Amount: ${profile_dict['total_purchase_amount']:.2f}

Purchase Request:
- Instruction: {instruction}"""
    
    def _format_agent_output(self, instruction: str, query: str) -> str:
        """Format agent output for GuardAgent (query kept in signature for callers; not shown to the model)."""
        return f"""User wants to purchase a product based on:
- Instruction: {instruction}

Please check if this purchase request violates any business rules."""

    @staticmethod
    def _message_content_to_strings(content) -> List[str]:
        """Normalize message content to plain strings (OpenAI string, list of parts, etc.)."""
        if content is None:
            return []
        if isinstance(content, str):
            return [content]
        if isinstance(content, list):
            parts: List[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    if item.get("type") == "text" and item.get("text") is not None:
                        parts.append(str(item["text"]))
                    elif "text" in item and item["text"] is not None:
                        parts.append(str(item["text"]))
            return parts
        return [str(content)]

    @staticmethod
    def _parse_function_arguments(argums) -> Optional[dict]:
        if isinstance(argums, dict):
            return argums
        if isinstance(argums, str):
            try:
                parsed = json.loads(argums)
                return parsed if isinstance(parsed, dict) else None
            except Exception:
                return None
        return None

    @staticmethod
    def _append_code_cell_from_arguments(chunks: List[str], arguments) -> None:
        parsed = WebShopGuardAgent._parse_function_arguments(arguments)
        if isinstance(parsed, dict) and "cell" in parsed:
            cell = parsed["cell"]
            chunks.append(cell if isinstance(cell, str) else str(cell))
        else:
            chunks.append(str(arguments))

    @staticmethod
    def _chunks_from_message_list(messages: Optional[List]) -> List[str]:
        """
        Extract text/code from an AutoGen / OpenAI-style message list.
        Covers: plain content, multimodal content list, function_call, tool_calls, tool role replies.
        """
        if not messages:
            return []
        chunks: List[str] = []
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            for text in WebShopGuardAgent._message_content_to_strings(msg.get("content")):
                chunks.append(text)

            fc = msg.get("function_call")
            if isinstance(fc, dict):
                WebShopGuardAgent._append_code_cell_from_arguments(
                    chunks, fc.get("arguments", "")
                )

            tcalls = msg.get("tool_calls")
            if isinstance(tcalls, list):
                for tc in tcalls:
                    if not isinstance(tc, dict):
                        continue
                    fn = tc.get("function")
                    if isinstance(fn, dict):
                        WebShopGuardAgent._append_code_cell_from_arguments(
                            chunks, fn.get("arguments", "")
                        )

        return chunks

    @staticmethod
    def _chunks_from_oai_messages(agent) -> List[str]:
        """Turn one agent's _oai_messages into ordered text chunks (content + code cells)."""
        logs = getattr(agent, "_oai_messages", None) or {}
        chunks: List[str] = []
        for peer in list(logs.keys()):
            chunks.extend(WebShopGuardAgent._chunks_from_message_list(logs[peer]))
        return chunks

    def _collect_guardagent_log_chunks(self) -> List[str]:
        """
        Merge chat transcripts from both GuardAgent and chatbot.
        Tool/function returns sometimes appear only in chat_messages, or only on one side of
        the AutoGen _oai_messages map — include both.
        """
        ordered: List[str] = []
        seen: set = set()
        ga, cb = self.guard_agent, self.chatbot

        def _add(lst: List[str]) -> None:
            for chunk in lst:
                if chunk in seen:
                    continue
                seen.add(chunk)
                ordered.append(chunk)

        _add(self._chunks_from_oai_messages(ga))
        _add(self._chunks_from_oai_messages(cb))
        cm_ga = getattr(ga, "chat_messages", None) or {}
        cm_cb = getattr(cb, "chat_messages", None) or {}
        _add(self._chunks_from_message_list(cm_ga.get(cb)))
        _add(self._chunks_from_message_list(cm_cb.get(ga)))
        return ordered

    _MARKDOWN_FENCE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)

    def _maybe_salvage_markdown_python(self, logs_string: List[str]) -> List[str]:
        """
        If chat logs contain ```python ... ``` but no GuardAgent tool return, run the cell through
        run_code_webshop (adds CodeHeader so CheckRule exists). Covers models that emit markdown
        instead of calling the python function after code_execution_config was disabled on UserProxy.
        """
        if any(isinstance(s, str) and "GuardAgent results:" in s for s in logs_string):
            return logs_string
        for section in reversed(logs_string):
            if not isinstance(section, str):
                continue
            matches = list(self._MARKDOWN_FENCE.finditer(section))
            for m in reversed(matches):
                cell = m.group(1).strip()
                if len(cell) < 25:
                    continue
                if "CheckRule" not in cell and "check_rule" not in cell:
                    continue
                try:
                    out = run_code_webshop(cell)
                    if isinstance(out, str) and out:
                        if self.verbose:
                            print("[GuardAgent] Salvaged markdown ```python``` via run_code_webshop")
                        return logs_string + [out]
                except Exception as e:
                    if self.verbose:
                        print(f"[GuardAgent] Markdown salvage failed: {e}")
                    return logs_string + [f"Salvage execution error: {e}"]
        return logs_string

    def _parse_guard_agent_results(self, logs_string: List[str]) -> Tuple[bool, List[str]]:
        """
        Parse GuardAgent results from logs
        Returns: (is_valid, violated_rules)
        
        IMPORTANT: If GuardAgent execution fails (no "GuardAgent results:" found),
        we return (False, ["guardagent_execution_failed"]) instead of defaulting to valid.
        This is a security-first approach: if we can't verify, we should block.
        """
        violated_rules = []
        is_valid = None  # None means we haven't found results yet
        found_results = False
        
        def _normalize_rule_name(name: str) -> str:
            return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")

        # Look for GuardAgent results in logs
        # Compatible formats:
        # 1) "GuardAgent results:\naction_denied: ..."
        # 2) "exitcode: ...\nCode output:\nACTION GRANTED|ACTION DENIED ..."
        for section in logs_string:
            if isinstance(section, str) and "GuardAgent results:" in section:
                found_results = True
                # Parse the results
                try:
                    # Extract action_denied
                    if "action_denied: 1" in section or "action_denied: True" in section:
                        is_valid = False

                        # Preferred source: inaccessible_actions (contains rule keys directly)
                        if "inaccessible_actions:" in section:
                            raw_inaccessible = section.split("inaccessible_actions:", 1)[1]
                            raw_inaccessible = raw_inaccessible.split("\nguardrailed_answer:", 1)[0].strip()
                            if raw_inaccessible and raw_inaccessible not in ("{}", "None"):
                                try:
                                    parsed = ast.literal_eval(raw_inaccessible)
                                    if isinstance(parsed, dict):
                                        for key in parsed.keys():
                                            if isinstance(key, str):
                                                violated_rules.append(_normalize_rule_name(key))
                                except Exception:
                                    # Fallback: parse keys from dict-like string
                                    for key in re.findall(r"'([^']+)'\s*:", raw_inaccessible):
                                        violated_rules.append(_normalize_rule_name(key))

                        # Backward-compatible fallback: infer from guardrailed_answer text
                        if not violated_rules and "guardrailed_answer:" in section:
                            guardrailed_answer = section.split("guardrailed_answer:", 1)[1]
                            guardrailed_answer = guardrailed_answer.split("\n", 1)[0].strip()
                            if guardrailed_answer:
                                violated_rules.append(_normalize_rule_name(guardrailed_answer))

                        if not violated_rules:
                            violated_rules.append("unknown_rule")
                    
                    elif "action_denied: 0" in section or "action_denied: False" in section:
                        is_valid = True
                        
                except Exception as e:
                    if self.verbose:
                        print(f"[Warning] Failed to parse GuardAgent results: {e}")
                        import traceback
                        traceback.print_exc()
                    # If parsing fails, we can't trust the result - default to invalid (security-first)
                    is_valid = False
                    violated_rules.append("guardagent_parse_error")

            # Compatibility: parse raw execution output format
            elif isinstance(section, str) and "Code output:" in section:
                section_lower = section.lower()
                if "action denied" in section_lower:
                    found_results = True
                    is_valid = False

                    # Try to extract violation details from common output patterns.
                    violation_text = ""
                    if "violation:" in section_lower:
                        violation_text = section.split("violation:", 1)[1].split("\n", 1)[0].strip()
                    elif "guardrail triggered:" in section_lower:
                        violation_text = section.split("guardrail triggered:", 1)[1].split("\n", 1)[0].strip()

                    if violation_text:
                        # Keep a compact normalized tag for metrics/logging.
                        normalized = _normalize_rule_name(violation_text)
                        violated_rules.append(normalized if normalized else "unknown_rule")
                    elif not violated_rules:
                        violated_rules.append("unknown_rule")

                elif "action granted" in section_lower:
                    found_results = True
                    # Preserve deny signal if already detected in another section.
                    if is_valid is None:
                        is_valid = True
        
        # If no results found, GuardAgent execution likely failed
        if not found_results:
            if self.verbose:
                print("[Error] GuardAgent execution failed: No 'GuardAgent results:' found in logs")
                print(f"[Debug] Logs string length: {len(logs_string)}")
                print(f"[Debug] Logs preview: {str(logs_string[:3]) if logs_string else 'Empty'}")
            # Security-first: if we can't verify, we should block
            is_valid = False
            violated_rules.append("guardagent_execution_failed")
        
        # If is_valid is still None (shouldn't happen, but safety check)
        if is_valid is None:
            if self.verbose:
                print("[Warning] GuardAgent results parsing returned None - defaulting to invalid")
            is_valid = False
            violated_rules.append("guardagent_unknown_error")
        
        return is_valid, violated_rules

    def _update_memory(self, agent_input: str, agent_output: str, logs_string: List[str]):
        """Update long-term memory with successful check"""
        try:
            # Extract subtasks and code from logs
            subtasks = None
            code = None
            
            for section in logs_string:
                if isinstance(section, str):
                    if "Task decomposition:" in section:
                        subtasks = section.split("Task decomposition:")[-1]
                        if "Guardrail code:" in subtasks:
                            subtasks = subtasks.split("Guardrail code:")[0]
                    if "Guardrail code:" in section:
                        code = section.split("Guardrail code:")[-1]
            
            # Get code from guard_agent
            if not code and hasattr(self.guard_agent, 'code'):
                code = self.guard_agent.code
            
            if subtasks and code:
                new_item = {
                    "agent input": agent_input,
                    "agent output": agent_output,
                    "subtasks": subtasks,
                    "code": code
                }
                self.long_term_memory.append(new_item)
                
                if self.verbose:
                    print(f"[GuardAgent] Updated memory (total items: {len(self.long_term_memory)})")
        except Exception as e:
            if self.verbose:
                print(f"[Warning] Failed to update memory: {e}")
