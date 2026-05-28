"""Configuration for FragFuse fusion-instruction optimization."""

import os
from typing import Any, Dict


class Config:
    """Runtime configuration for FragFuse optimization experiments."""

    def __init__(self):
        self.population_size = 20
        self.num_generations = 50
        self.elite_size = 3

        self.llm_rewrite_variants = 5
        self.symbol_proposer_variants_per_template = 3
        self.crossover_rate = 0.3
        self.mutation_rate = 0.1
        self.max_prompt_length = 200

        self.llm_config = {
            "model": "gpt-4o",
            "temperature": 0.8,
            "max_tokens": 150,
            "api_base": "http://152.53.53.64:3000/v1",
        }

        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.results_dir = os.path.join(self.base_dir, "results")

        self.experiment_id = self._get_next_experiment_id()
        self.experiment_dir = os.path.join(self.results_dir, f"optimization_{self.experiment_id}")
        os.makedirs(self.experiment_dir, exist_ok=True)

        self.best_fusion_instructions_file = os.path.join(
            self.experiment_dir, "best_fusion_instructions.json"
        )
        self.optimization_log_file = os.path.join(self.experiment_dir, "optimization_log.txt")
        self.optimization_log_full_file = os.path.join(
            self.experiment_dir, "optimization_log_full.txt"
        )
        self.population_history_file = os.path.join(
            self.experiment_dir, "population_history.json"
        )

        self.request_interval = 0.2
        self.no_improvement_generations = 10
        self.lambda_ret = 0.2

    def to_dict(self) -> Dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "population_size": self.population_size,
            "num_generations": self.num_generations,
            "elite_size": self.elite_size,
            "llm_rewrite_variants": self.llm_rewrite_variants,
            "symbol_proposer_variants_per_template": self.symbol_proposer_variants_per_template,
            "crossover_rate": self.crossover_rate,
            "mutation_rate": self.mutation_rate,
            "max_prompt_length": self.max_prompt_length,
            "llm_config": self.llm_config,
            "no_improvement_generations": self.no_improvement_generations,
            "lambda_ret": self.lambda_ret,
        }

    def update_from_dict(self, config_dict: Dict[str, Any]) -> None:
        for key, value in config_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def save_to_file(self, file_path: str) -> None:
        import json

        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    def load_from_file(self, file_path: str) -> None:
        import json

        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                self.update_from_dict(json.load(f))

    def _get_next_experiment_id(self) -> int:
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir, exist_ok=True)
            return 1

        existing_experiments = []
        for item in os.listdir(self.results_dir):
            path = os.path.join(self.results_dir, item)
            if os.path.isdir(path) and item.startswith("optimization_"):
                try:
                    existing_experiments.append(int(item.split("_")[1]))
                except (ValueError, IndexError):
                    continue

        return max(existing_experiments) + 1 if existing_experiments else 1

    def __str__(self) -> str:
        return (
            "FragFuse Optimization Config:\n"
            f"  Experiment ID: {self.experiment_id}\n"
            f"  Population Size: {self.population_size}\n"
            f"  Generations: {self.num_generations}\n"
            f"  Elite Size: {self.elite_size}\n"
            f"  LLM Model: {self.llm_config['model']}\n"
            f"  Scoring: score = fusion_success_score - {self.lambda_ret} * L_ret"
        )
