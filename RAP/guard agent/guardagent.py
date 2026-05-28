import time
from typing import Callable, Dict, List, Literal, Optional, Union
import logging
import openai
import json
from openai import OpenAI
from autogen.agentchat import Agent, UserProxyAgent
import Levenshtein

logger = logging.getLogger(__name__)


def _guardagent_chat_completion(config: dict, messages: list) -> str:
    """
    Run one chat-style completion for GuardAgent internals.

    Uses Google GenAI when ``config['use_gemini_client']`` is True (same relay as main.py /
    RuleChecker); otherwise OpenAI-compatible ``chat.completions``.
    """
    if config.get("use_gemini_client"):
        import os as _os

        gkey = config.get("gemini_api_key")
        if not gkey:
            raise ValueError("Gemini defense enabled but gemini_api_key missing in config.")
        gbase = config.get("gemini_base_url", "http://148.113.224.153:3000")
        gmodel = config.get("gemini_model") or "gemini-2.5-flash"
        _os.environ["GEMINI_API_KEY"] = gkey
        try:
            from google import genai
        except ImportError as e:
            raise ImportError(
                "google-genai is required for Gemini GuardAgent. pip install google-genai"
            ) from e
        client = genai.Client(http_options={"base_url": gbase})
        chunks = [f"{m.get('role', 'user').upper()}:\n{m.get('content', '')}" for m in messages]
        prompt = "\n\n".join(chunks)
        response = client.models.generate_content(model=gmodel, contents=prompt)
        text = getattr(response, "text", None)
        if text and str(text).strip():
            return str(text).strip()
        return str(response).strip()

    api_base = config.get("api_base") or config.get("base_url", None)
    engine = config["model"]
    if api_base:
        client = OpenAI(api_key=config["api_key"], base_url=api_base)
    else:
        client = OpenAI(api_key=config["api_key"])
    response = client.chat.completions.create(
        model=engine,
        messages=messages,
        temperature=0,
        max_tokens=1000,
        top_p=0.95,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None,
    )
    return response.choices[0].message.content.strip()


class GuardAgent(UserProxyAgent):
    def __init__(
            self,
            name: str,
            is_termination_msg: Optional[Callable[[Dict], bool]] = None,
            max_consecutive_auto_reply: Optional[int] = None,
            human_input_mode: Optional[str] = "ALWAYS",
            function_map: Optional[Dict[str, Callable]] = None,
            code_execution_config: Optional[Union[Dict, Literal[False]]] = None,
            default_auto_reply: Optional[Union[str, Dict, None]] = "",
            llm_config: Optional[Union[Dict, Literal[False]]] = False,
            system_message: Optional[Union[str, List]] = "",
            config_list: Optional[List[Dict]] = None,
    ):
        super().__init__(
            name=name,
            system_message=system_message,
            is_termination_msg=is_termination_msg,
            max_consecutive_auto_reply=max_consecutive_auto_reply,
            human_input_mode=human_input_mode,
            function_map=function_map,
            code_execution_config=code_execution_config,
            llm_config=llm_config,
            default_auto_reply=default_auto_reply,
        )
        self.config_list = config_list
        self.user_request = ''
        self.agent_specification = ''
        self.agent_input = ''
        self.agent_output = ''
        self.subtasks = ''
        self.code = ''
        # Debug trace for two-stage GuardAgent pipeline logging.
        self.last_task_decomposition_prompt = ''
        self.last_task_decomposition_output = ''
        self.last_codegen_prompt = ''

    def task_decomposition(self, config, user_request, agent_specification, agent_input, agent_output, Decomposition_Examples):
        # import prompt
        from prompts_guard import Example_Decomposition
        # Returns the related information to the given query.
        patience = 2
        sleep_time = 30
        if not config.get("use_gemini_client"):
            openai.api_key = config["api_key"]
        query_message = Example_Decomposition.format(user_request=user_request,
                                                     agent_specification=agent_specification,
                                                     decomposition_examples=Decomposition_Examples,
                                                     agent_input=agent_input,
                                                     agent_output=agent_output)
        self.last_task_decomposition_prompt = query_message
        from prompts_guard import SYSTEM_PROMPT_DECOMPOSITION
        messages = [{"role": "system", "content": SYSTEM_PROMPT_DECOMPOSITION},
                    {"role": "user", "content": query_message}]
        while patience > 0:
            patience -= 1
            try:
                prediction = _guardagent_chat_completion(config, messages)
                if prediction != "" and prediction is not None:
                    self.last_task_decomposition_output = prediction
                    return prediction
            except Exception as e:
                logger.warning("GuardAgent task decomposition failed: %s", e)
                if sleep_time > 0:
                    time.sleep(sleep_time)
        return "Fail to retrieve related knowledge, please try again later."

    def retrieve_examples(self, agent_input, agent_output):
        # Validate memory and num_shots
        if not hasattr(self, 'memory') or self.memory is None:
            self.memory = []  # Initialize empty memory if not set
        if not isinstance(self.memory, list):
            raise ValueError(f"self.memory must be a list. Got: {type(self.memory)}")
        if not hasattr(self, 'num_shots') or self.num_shots is None:
            self.num_shots = 0  # Default to 0 if not set
        
        # If memory is empty, return empty examples
        if len(self.memory) == 0:
            return ""
        
        levenshtein_dist = {}
        for i in range(len(self.memory)):
            # Validate memory item structure
            if not isinstance(self.memory[i], dict):
                continue  # Skip invalid items
            required_keys = ["agent input", "agent output", "subtasks", "code"]
            if not all(key in self.memory[i] for key in required_keys):
                continue  # Skip items with missing keys
            
            mem_input = self.memory[i]["agent input"]
            mem_output = self.memory[i]["agent output"]
            levenshtein_dist[i] = Levenshtein.distance(agent_input, mem_input) + Levenshtein.distance(agent_output, mem_output)
        
        if len(levenshtein_dist) == 0:
            return ""  # No valid examples found
        
        levenshtein_dist = sorted(levenshtein_dist.items(), key=lambda x: x[1], reverse=False)
        selected_indexes = [levenshtein_dist[i][0] for i in range(min(self.num_shots, len(levenshtein_dist)))]
        examples = []
        for i in selected_indexes:
            template = "Agent input:\n {}\nAgent output:\n{}\nTask decomposition:\n{}\nGuardrail code:\n{}\n".format(self.memory[i]["agent input"],
                                                                                                               self.memory[i]["agent output"],
                                                                                                               self.memory[i]["subtasks"],
                                                                                                               self.memory[i]["code"])
            examples.append(template)
        examples = '\n'.join(examples)
        return examples

    def generate_init_message(self, **context):
        # Validate required context keys
        required_keys = ["user_request", "agent_specification", "agent_input", "agent_output", "agent_task_deco_examples"]
        missing_keys = [key for key in required_keys if key not in context]
        if missing_keys:
            raise KeyError(f"Missing required keys in context: {missing_keys}")

        self.user_request = context["user_request"]
        self.agent_specification = context["agent_specification"]
        self.agent_input = context["agent_input"]
        self.agent_output = context["agent_output"]
        self.agent_task_deco_examples = context["agent_task_deco_examples"]

        # Validate config_list
        if self.config_list is None:
            raise ValueError("config_list is None. GuardAgent must be initialized with config_list.")
        if not isinstance(self.config_list, list) or len(self.config_list) == 0:
            raise ValueError(f"config_list must be a non-empty list. Got: {type(self.config_list)}")
        if not isinstance(self.config_list[0], dict):
            raise ValueError(f"config_list[0] must be a dict. Got: {type(self.config_list[0])}")
        cfg0 = self.config_list[0]
        if cfg0.get("use_gemini_client"):
            if not cfg0.get("gemini_api_key") or not cfg0.get("gemini_model"):
                raise KeyError(
                    "config_list[0] for Gemini must contain 'gemini_api_key' and 'gemini_model'. "
                    f"Got keys: {list(cfg0.keys())}"
                )
        elif "api_key" not in cfg0 or "model" not in cfg0:
            raise KeyError(
                f"config_list[0] must contain 'api_key' and 'model' keys. Got keys: {list(cfg0.keys())}"
            )

        # import prompt
        from prompts_guard import GuardAgent_Message_Prompt
        subtasks = self.task_decomposition(self.config_list[0],
                                           self.user_request,
                                           self.agent_specification,
                                           self.agent_input,
                                           self.agent_output,
                                           self.agent_task_deco_examples)
        self.subtasks = subtasks

        examples = self.retrieve_examples(self.agent_input, self.agent_output)

        init_message = GuardAgent_Message_Prompt.format(examples=examples,
                                                        agent_input=self.agent_input,
                                                        agent_output=self.agent_output,
                                                        subtasks=subtasks)
        self.last_codegen_prompt = init_message
        return init_message

    def send(self, message: Union[Dict, str], recipient: Agent, request_reply: Optional[bool] = None,
             silent: Optional[bool] = False):
        # Some OpenAI-compatible backends reject assistant messages with content=null.
        # Normalize to empty string when function_call is present.
        if isinstance(message, dict) and message.get("content", None) is None:
            message = dict(message)
            message["content"] = ""
        valid = self._append_oai_message(message, "assistant", recipient)
        if valid:
            recipient.receive(message, self, request_reply, silent)
        else:
            raise ValueError(
                "Message can't be converted into a valid ChatCompletion message. Either content or function_call must be provided."
            )

    def initiate_chat(self, recipient: "ConversableAgent", clear_history: Optional[bool] = True,
                      silent: Optional[bool] = False, **context, ):
        self._prepare_chat(recipient, clear_history)
        self.send(self.generate_init_message(**context), recipient, silent=silent)

    def receive(
            self,
            message: Union[Dict, str],
            sender: Agent,
            request_reply: Optional[bool] = None,
            silent: Optional[bool] = False,
    ):
        self._process_received_message(message, sender, silent)
        if request_reply is False or request_reply is None and self.reply_at_receive[sender] is False:
            return
        # Some backends reject chat history items with content=null.
        # Autogen may keep function_call messages with content=None in history.
        for msg in self.chat_messages.get(sender, []):
            if isinstance(msg, dict) and msg.get("content", None) is None:
                msg["content"] = ""

        reply = self.generate_reply(messages=self.chat_messages[sender], sender=sender)
        if reply is not None:
            # We only need to return function execution results to the assistant.
            # Prevent an extra auto-reply round, which can include legacy
            # function-call history entries with content=None on strict backends.
            self.send(reply, sender, request_reply=False, silent=silent)

    def error_debugger(self, config, code, error_info):
        # import prompt
        from prompts_guard import CodeDebugger
        # Returns the related information to the given query.
        patience = 1
        sleep_time = 30
        if not config.get("use_gemini_client"):
            openai.api_key = config["api_key"]
        query_message = CodeDebugger.format(subtasks=self.subtasks, code=code, error_info=error_info)
        messages = [{"role": "system",
                     "content": "You are an AI assistant that helps people debug their code. Only list one most possible reason to the errors."},
                    {"role": "user", "content": query_message}]
        while patience > 0:
            patience -= 1
            try:
                prediction = _guardagent_chat_completion(config, messages)
                if prediction != "" and prediction is not None:
                    return prediction
            except Exception as e:
                logger.warning("GuardAgent error debugger failed: %s", e)
                if sleep_time > 0:
                    time.sleep(sleep_time)
        return "Fail to diagnose the reasons to the errors."

    def execute_function(self, func_call):
        """Execute a function call and return the result.

        Override this function to modify the way to execute a function call.

        Args:
            func_call: a dictionary extracted from openai message at key "function_call" with keys "name" and "arguments".

        Returns:
            A tuple of (is_exec_success, result_dict).
            is_exec_success (boolean): whether the execution is successful.
            result_dict: a dictionary with keys "name", "role", and "content". Value of "role" is "function".
        """
        func_name = func_call.get("name", "")
        func = self._function_map.get(func_name, None)

        is_exec_success = False
        if func is not None:
            # Extract arguments from a json-like string and put it into a dict.
            input_string = self._format_json_str(func_call.get("arguments", "{}"))
            try:
                arguments = json.loads(input_string)
            except json.JSONDecodeError as e:
                arguments = None
                if func_call["arguments"].find('\"\"\"') > 0:
                    arguments_string = func_call["arguments"].split('\"\"\"')[1]
                else:
                    arguments_string = func_call["arguments"].split(': "')[-1]
                    arguments_string = arguments_string.split('", ')[0]
                arguments = {"cell": arguments_string}
                # content = f"Error: {e}\n You argument should follow json format."
                content = f"Error: {e}\n There might be compilation errors in the code. Please check the code and try again."

            # Try to execute the function
            if arguments is not None:
                self.code = arguments["cell"]
                try:
                    content = func(**arguments)
                    is_exec_success = True
                except Exception as e:
                    content = f"Error: {e}"
        else:
            content = f"Error: Function {func_name} not found."
        if "error" in content or "Error" in content:
            reasons = self.error_debugger(self.config_list[0], self.code, content)
            content = content + '\nPotential Reasons: ' + reasons

        return is_exec_success, {
            "name": func_name,
            "role": "function",
            "content": str(content),
        }

    def update_memory(self, num_shots, memory):
        self.num_shots = num_shots
        self.memory = memory

