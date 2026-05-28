"""
WebShop-specific code execution function for GuardAgent
"""

import re
import sys
import traceback
from io import StringIO


def _fix_missing_def_keyword(cell: str) -> str:
    """
    Models sometimes emit 'define_name():' or 'define name():' instead of 'def ...'.
    Only rewrite line-start patterns that look like a mistaken function header.
    """
    if not cell:
        return cell
    # define_user_profile(): -> def define_user_profile():
    cell = re.sub(
        r"(?m)^(\s*)define_([a-zA-Z][a-zA-Z0-9_]*)\s*(\([^)]*\))\s*:",
        r"\1def define_\2\3:",
        cell,
    )
    # define foo(): -> def foo():
    cell = re.sub(
        r"(?m)^(\s*)define\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*(\([^)]*\))\s*:",
        r"\1def \2\3:",
        cell,
    )
    return cell


def _fix_triple_single_trailing_quote_bug(cell: str) -> str:
    """
    Inside '''...''', models often end Disregard '...word' then close with ''' but write
    ...word'''' (no space). Python then mis-tokenizes → unterminated string.
    Normalize ...word'''' → ...word' ''' (same fix as a trailing apostrophe + closing delimiter).
    """
    if not cell or "''''" not in cell:
        return cell
    # Word/number char followed by four quotes: almost always word' + ''' mangled together.
    return re.sub(r"([a-zA-Z0-9])''''", r"\1' '''", cell)


def _infer_outcome_from_return_dict(res: dict):
    """
    Models often end with result = guardrail_code(...); result without print().
    If res has an 'action' field (GRANTED/DENIED variants), map to pipeline fields.
    """
    if not isinstance(res, dict):
        return None
    action = res.get("action")
    if action is None:
        action = res.get("ACTION") or res.get("outcome")
    if action is None:
        return None
    s = str(action).strip().upper()
    if "DENIED" in s or s in ("DENY", "REJECT", "BLOCK"):
        viol = res.get("violation", res.get("violations", ""))
        if isinstance(viol, dict):
            viol = str(viol)
        else:
            viol = str(viol) if viol is not None else ""
        return True, viol, ""
    if "GRANTED" in s or s in ("GRANT", "ALLOW", "OK", "PASS", "APPROVE"):
        ga = res.get("guardrailed_answer", res.get("answer", ""))
        if isinstance(ga, dict):
            ga = str(ga)
        else:
            ga = str(ga) if ga is not None else ""
        return False, "", ga
    return None


def _infer_outcome_from_exec_globals(global_var: dict):
    for key in ("result", "guardrail_result", "check_result"):
        t = _infer_outcome_from_return_dict(global_var.get(key))
        if t is not None:
            return t
    return None


def run_code_webshop(cell):
    """
    Execute GuardAgent generated code for WebShop rule checking.
    Returns formatted results similar to run_code_seeact.
    """
    from prompts_guard import CodeHeader
    try:
        # Models often wrap logic in def guardrail_code() and only call it under
        # if __name__ == "__main__". Plain exec() otherwise sees __name__ as "builtins",
        # so the branch never runs and stdout has no ACTION lines.
        global_var = {
            "__name__": "__main__",
            "action_denied": None,
            "inaccessible_actions": None,
            "guardrailed_answer": None,
        }
        cell = _fix_missing_def_keyword(cell)
        cell = _fix_triple_single_trailing_quote_bug(cell)

        old_stdout = sys.stdout
        redirected_output = StringIO()
        try:
            sys.stdout = redirected_output
            exec(CodeHeader + cell, global_var)
        finally:
            sys.stdout = old_stdout
        output = redirected_output.getvalue()
        out_lo = output.lower()
        
        # Parse output to determine action_denied
        action_denied = False
        guardrailed_answer = ""
        inaccessible_actions = ""
        
        if "action denied" in out_lo:
            action_denied = True
            idx = out_lo.find("violation:")
            if idx >= 0:
                violation_part = output[idx + len("violation:") :].split("\n")[0].strip()
                inaccessible_actions = violation_part
        elif "action granted" in out_lo:
            action_denied = False
            idx = out_lo.find("guardrailed_answer:")
            if idx >= 0:
                guardrailed_answer = output[idx + len("guardrailed_answer:") :].split("\n")[0].strip()
        else:
            # Stdout missing keywords: infer from exec namespace (models often assign but forget to print)
            gv_ad = global_var.get("action_denied")
            if gv_ad is not None:
                action_denied = bool(gv_ad)
                viol = global_var.get("violation")
                if viol is not None:
                    inaccessible_actions = viol if isinstance(viol, str) else str(viol)
                ga = global_var.get("guardrailed_answer")
                if ga is not None and ga != "":
                    guardrailed_answer = str(ga)

        outcome_from_dict = False
        if (
            "action denied" not in out_lo
            and "action granted" not in out_lo
            and global_var.get("action_denied") is None
        ):
            inferred = _infer_outcome_from_exec_globals(global_var)
            if inferred is not None:
                action_denied, inaccessible_actions, guardrailed_answer = inferred
                outcome_from_dict = True

        # Validate: stdout keywords, top-level action_denied, or recognized return dict (e.g. result = {...})
        if (
            "action denied" not in out_lo
            and "action granted" not in out_lo
            and global_var.get("action_denied") is None
            and not outcome_from_dict
        ):
            return "Missing variables. Code must print either 'ACTION DENIED' or 'ACTION GRANTED'."
        
        return "GuardAgent results:\naction_denied: {}\ninaccessible_actions: {}\nguardrailed_answer: {}\n(End of results)".format(
            int(action_denied), inaccessible_actions, guardrailed_answer
        )
    except Exception as e:
        error_info = traceback.format_exc()
        code = CodeHeader + cell
        
        # Parse error information
        if "SyntaxError" in str(repr(e)):
            error_line = str(repr(e))
            error_type = error_line.split('(')[0]
            error_message = error_line.split(',')[0].split('(')[1]
            error_line = error_line.split('"')[1] if '"' in error_line else ""
        elif "KeyError" in str(repr(e)):
            code_lines = code.split('\n')
            key = str(repr(e)).split("'")[1]
            error_type = str(repr(e)).split('(')[0]
            error_line = ""
            for i in range(len(code_lines)):
                if key in code_lines[i]:
                    error_line = code_lines[i]
                    break
            error_message = str(repr(e))
        elif "TypeError" in str(repr(e)):
            error_type = str(repr(e)).split('(')[0]
            error_message = str(e)
            function_mapping_dict = {
                "check_access": "CheckAccess",
                "check_rule": "CheckRule"
            }
            error_key = ""
            for key in function_mapping_dict.keys():
                if key in error_message:
                    error_message = error_message.replace(key, function_mapping_dict[key])
                    error_key = function_mapping_dict[key]
            code_lines = code.split('\n')
            error_line = ""
            for i in range(len(code_lines)):
                if error_key in code_lines[i]:
                    error_line = code_lines[i]
                    break
        else:
            error_type = ""
            error_message = str(repr(e)).split("('")[-1].split("')")[0] if "('" in str(repr(e)) else str(e)
            error_line = ""
        
        # Format error information
        if error_type != "" and error_line != "":
            error_info = f'{error_type}: {error_message}. The error messages occur in the code line "{error_line}".'
        else:
            error_info = f'Error: {error_message}.'
        error_info += '\nPlease make modifications accordingly and make sure the rest code works well with the modification.'
        
        return error_info
