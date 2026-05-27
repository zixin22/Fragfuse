# cd ..
# python DAS/exp_OS.py --dataset benign --memory memory_os_benign_claude.json --agent_model gpt-5-2025-08-07 --use_memory --guardrail_model gpt-4o --use_guardrail
# python DAS/exp_OS.py --dataset system_sabotage --memory memory_os_benign_claude.json --agent_model gpt-5-2025-08-07 --use_memory --guardrail_model gpt-4o --use_guardrail
# python DAS/exp_OS.py --dataset prompt_injection --memory memory_os_benign_claude.json --agent_model gpt-4-turbo --guardrail_model gpt-4o --use_guardrail
# python DAS/exp_OS.py --dataset environment --memory memory_os_benign_claude.json --agent_model gpt-4o --guardrail_model gpt-4o --use_guardrail

python DAS/exp_OS.py --dataset benign --agent_model gpt-4o
python DAS/exp_OS.py --dataset benign --memory memory_os_benign_claude.json --agent_model claude-sonnet-4-5-20250929 --guardrail_model claude-sonnet-4-5-20250929 --use_guardrail --no_memory --only_guard

python DAS/exp_OS.py --dataset system_sabotage --memory memory_os_benign_claude.json --agent_model gpt-4o --guardrail_model gpt-4o --use_guardrail --no_memory --only_guard
python DAS/exp_OS.py --dataset system_sabotage --memory memory_os_benign_claude.json --agent_model gpt-5.1 --guardrail_model gpt-5.1 --use_guardrail --no_memory --only_guard
python DAS/exp_OS.py --dataset system_sabotage --memory memory_os_benign_claude.json --agent_model claude-sonnet-4-5-20250929 --guardrail_model claude-sonnet-4-5-20250929 --use_guardrail --no_memory --only_guard
python DAS/exp_OS.py --dataset system_sabotage --memory memory_os_benign_claude.json --agent_model gemini-2.5-flash --guardrail_model gemini-2.5-flash --use_guardrail --no_memory --only_guard

python DAS/exp_OS.py --dataset environment --memory memory_os_benign_claude.json --agent_model gpt-4o --guardrail_model gpt-4o --use_guardrail --no_memory --only_guard
python DAS/exp_OS.py --dataset environment --memory memory_os_benign_claude.json --agent_model gpt-4o --guardrail_model gpt-5.1 --use_guardrail --no_memory --only_guard
python DAS/exp_OS.py --dataset environment --memory memory_os_benign_claude.json --agent_model gpt-4o --guardrail_model claude-sonnet-4-5-20250929 --use_guardrail --no_memory --only_guard
python DAS/exp_OS.py --dataset environment --memory memory_os_benign_claude.json --agent_model gpt-4o --guardrail_model gemini-2.5-flash  --use_guardrail --no_memory --only_guard

python DAS/exp_OS.py --dataset system_sabotage --memory memory_os_benign_claude.json --agent_model gpt-4o --use_memory
python DAS/exp_OS.py --dataset environment --memory memory_os_benign_claude.json --agent_model gpt-4o --use_memory