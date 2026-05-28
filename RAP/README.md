# Get Started

## Responsible Use Notice

This repository is released only for research reproducibility and defensive evaluation. The code and prompts are intended to help study and mitigate memory-based access-control risks in LLM agents, not to enable misuse.

Please use these artifacts only in controlled benchmark or synthetic research settings. Do not apply them to real-world deployed systems or third-party agents without authorization.

We also provide mitigation guidance, including memory admission control, retrieval-time policy checking, and access-control enforcement after memory augmentation.

1. Enter the experiment directory and install requirements:

```bash
cd webshop
pip install -r requirements.txt
```

2. Follow the [WebShop setup instructions](https://github.com/princeton-nlp/WebShop?tab=readme-ov-file#-setup) to download the data, set up the webserver, and configure the required environment variables.

   Note: host the webserver and all data on localhost because the WebShop-provided link (`http://3.83.245.205:3000`) is no longer valid.

3. Run the webserver in a separate terminal. Stop and restart the webserver before each experiment.

```bash
cd WebShop-master
./run_dev.sh
```

4. Prepare your OpenAI API key and put it in `webshop/OpenAI_api_key.txt`.

You can use `webshop/fragExtracor_pipeline/output.json` as an example to generate your own `dataset_attack.json`.

5. Run experiment commands from the `webshop/` directory:

Attack:

```bash
python main.py --model gpt-4o --attack --dataset dataset_attack.json --limit 5 --retrieve_mode rap --output output/attack_smoke
```

RuleChecker:

```bash
python main.py --model gpt-4o --attack --dataset dataset_attack.json --defense_mode rule_checker --defense_check_target instruction --retrieve_mode rap --limit 10 --output output/rulechecker_test
```

GuardAgent:

```bash
python main.py --model gpt-4o --attack --dataset dataset_attack.json --defense_mode guard_agent --defense_check_target instruction --retrieve_mode rap --limit 10 --output output/guardagent_test
```
