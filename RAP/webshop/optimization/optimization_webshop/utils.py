"""Utility functions for FragFuse optimization experiments."""

import json
import os
from datetime import datetime
from typing import Any, Dict, List, Tuple

try:
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("[Warning] matplotlib not available. Plotting functions disabled.")


def load_seed_prompts(file_path: str) -> List[str]:
    """Load blank-line-separated fusion-instruction seeds."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Seed file not found: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read().strip()

    prompts = []
    for block in [b.strip() for b in content.split("\n\n") if b.strip()]:
        lines = [line.strip() for line in block.split("\n") if line.strip()]
        if lines:
            prompts.append("\n".join(lines))
    return prompts


def save_population_snapshot(population, generation: int, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    snapshot_file = os.path.join(output_dir, f"population_gen_{generation:03d}.json")
    snapshot = {
        "generation": generation,
        "timestamp": datetime.now().isoformat(),
        "population_size": len(population.members),
        "members": [ind.to_dict() for ind in population.members],
        "statistics": population.get_statistics(),
        "best_individual": population.get_best_individual().to_dict()
        if population.get_best_individual()
        else None,
    }
    with open(snapshot_file, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2, ensure_ascii=False)


def calculate_diversity_score(population) -> float:
    if len(population.members) <= 1:
        return 0.0
    prompts = [ind.prompt for ind in population.members]
    return len(set(prompts)) / len(prompts)


def plot_optimization_progress(log_file: str, output_file: str = None) -> None:
    if not HAS_MATPLOTLIB:
        print("[Warning] matplotlib not available. Cannot generate plots.")
        return
    if not os.path.exists(log_file):
        print(f"Optimization log not found: {log_file}")
        return

    with open(log_file, "r", encoding="utf-8") as f:
        log_data = json.load(f)

    generations, best_scores, avg_scores, diversities = [], [], [], []
    for entry in log_data:
        gen = entry.get("generation")
        if not isinstance(gen, int):
            continue
        stats = entry.get(
            "statistics_fusion_instruction",
            entry.get("statistics", {}),
        )
        generations.append(gen)
        best_scores.append(stats.get("max_score", 0))
        avg_scores.append(stats.get("avg_score", 0))
        diversities.append(stats.get("diversity", 0))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    ax1.plot(generations, best_scores, "r-", label="Best Score", linewidth=2)
    ax1.plot(generations, avg_scores, "b-", label="Average Score", alpha=0.7)
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Score")
    ax1.set_title("Optimization Progress")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(generations, diversities, "g-", label="Population Diversity", linewidth=2)
    ax2.set_xlabel("Generation")
    ax2.set_ylabel("Diversity")
    ax2.set_title("Population Diversity Over Generations")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        print(f"Saved progress plot: {output_file}")
    else:
        plt.show()


def _fusion_success_score(individual: Dict[str, Any]) -> float:
    return individual.get(
        "fusion_success_score",
        individual.get("score", 0.0),
    )


def analyze_optimization_results(results_file: str) -> Dict[str, Any]:
    if not os.path.exists(results_file):
        raise FileNotFoundError(f"Results file not found: {results_file}")

    with open(results_file, "r", encoding="utf-8") as f:
        results = json.load(f)

    best_individuals = results.get("best_individuals", [])
    if not best_individuals:
        return {}

    scores = [ind.get("score", 0.0) for ind in best_individuals]
    fusion_success_scores = [_fusion_success_score(ind) for ind in best_individuals]
    mean_score = sum(scores) / len(scores)

    return {
        "total_generations": results.get("total_generations", 0),
        "num_best_individuals": len(best_individuals),
        "score_distribution": {
            "mean": mean_score,
            "max": max(scores),
            "min": min(scores),
            "std": (sum((x - mean_score) ** 2 for x in scores) / len(scores)) ** 0.5,
        },
        "fusion_success_score_distribution": {
            "mean": sum(fusion_success_scores) / len(fusion_success_scores),
            "max": max(fusion_success_scores),
            "min": min(fusion_success_scores),
        },
        "final_best_prompt": (results.get("final_best") or {}).get("prompt", ""),
        "final_best_score": (results.get("final_best") or {}).get("score", 0.0),
    }


def print_optimization_summary(results_file: str) -> None:
    try:
        analysis = analyze_optimization_results(results_file)
        if not analysis:
            print("No optimization results found.")
            return

        print("=" * 60)
        print("FragFuse Optimization Summary")
        print("=" * 60)
        print(f"Generations: {analysis['total_generations']}")
        print(f"Best fusion instructions: {analysis['num_best_individuals']}")

        score_dist = analysis["score_distribution"]
        print("\nScore distribution:")
        print(f"  Mean: {score_dist['mean']:.3f}")
        print(f"  Max:  {score_dist['max']:.3f}")
        print(f"  Min:  {score_dist['min']:.3f}")
        print(f"  Std:  {score_dist['std']:.3f}")

        fusion_dist = analysis["fusion_success_score_distribution"]
        print("\nFusion success score distribution:")
        print(f"  Mean: {fusion_dist['mean']:.3f}")
        print(f"  Max:  {fusion_dist['max']:.3f}")
        print(f"  Min:  {fusion_dist['min']:.3f}")

        print("\nFinal best fusion instruction:")
        print(f"  Score: {analysis['final_best_score']:.3f}")
        print(f"  Instruction: {analysis['final_best_prompt']}")
        print("=" * 60)
    except Exception as e:
        print(f"Failed to summarize optimization results: {e}")


def compare_prompts(prompts: List[str]) -> Dict[str, Any]:
    comparison = {
        "num_prompts": len(prompts),
        "avg_length": sum(len(p) for p in prompts) / len(prompts) if prompts else 0.0,
        "unique_words": set(),
        "common_patterns": [],
    }
    for prompt in prompts:
        comparison["unique_words"].update(prompt.lower().split())
    comparison["unique_words"] = len(comparison["unique_words"])

    for pattern in ["extract", "find", "locate", "product", "spot", "<>", "[]"]:
        count = sum(1 for p in prompts if pattern in p.lower())
        if prompts and count > len(prompts) * 0.5:
            comparison["common_patterns"].append((pattern, count))
    return comparison


def export_prompts_to_file(
    prompts: List[str], output_file: str, format_type: str = "json"
) -> None:
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    if format_type == "json":
        data = {
            "exported_at": datetime.now().isoformat(),
            "num_prompts": len(prompts),
            "prompts": prompts,
        }
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    elif format_type == "txt":
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("# FragFuse Generated Fusion Instructions\n")
            f.write(f"# Generated at: {datetime.now().isoformat()}\n")
            f.write(f"# Total instructions: {len(prompts)}\n\n")
            for i, prompt in enumerate(prompts, 1):
                f.write(f"## Fusion Instruction {i}\n{prompt}\n\n")
    else:
        raise ValueError(f"Unsupported export format: {format_type}")
    print(f"Exported {len(prompts)} prompts to {output_file}")


def load_prompts_from_file(input_file: str) -> List[str]:
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Prompt file not found: {input_file}")

    file_ext = os.path.splitext(input_file)[1].lower()
    if file_ext == ".json":
        with open(input_file, "r", encoding="utf-8") as f:
            return json.load(f).get("prompts", [])
    if file_ext == ".txt":
        prompts, current_prompt = [], []
        with open(input_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("## Prompt") or line.startswith("## Fusion Instruction"):
                    if current_prompt:
                        prompts.append("\n".join(current_prompt))
                        current_prompt = []
                elif line and not line.startswith("#"):
                    current_prompt.append(line)
        if current_prompt:
            prompts.append("\n".join(current_prompt))
        return prompts
    raise ValueError(f"Unsupported prompt file extension: {file_ext}")


def setup_experiment_directory(base_dir: str, experiment_name: str) -> str:
    experiment_dir = os.path.join(base_dir, "experiments", experiment_name)
    for subdir in ["results", "logs", "plots"]:
        os.makedirs(os.path.join(experiment_dir, subdir), exist_ok=True)
    return experiment_dir


def validate_prompt(prompt: str) -> Tuple[bool, str]:
    if not prompt or len(prompt.strip()) < 5:
        return False, "Prompt is too short"
    if len(prompt) > 500:
        return False, "Prompt is too long"

    prompt_lower = prompt.lower()
    missing_elements = [
        elem for elem in ["spot", "product"] if elem not in prompt_lower
    ]
    if missing_elements:
        return False, f"Missing required elements: {missing_elements}"

    for pattern in ["hack", "exploit", "attack", "malware", "virus", "bypass"]:
        if pattern in prompt_lower:
            return False, f"Blocked unsafe term: {pattern}"
    return True, ""


def batch_validate_prompts(prompts: List[str]) -> List[Tuple[bool, str]]:
    return [validate_prompt(prompt) for prompt in prompts]


def generate_experiment_report(results_dir: str, output_file: str = None) -> str:
    if output_file is None:
        output_file = os.path.join(results_dir, "experiment_report.md")

    results_name = "best_fusion_instructions.json"
    results_file = os.path.join(results_dir, results_name)
    report_lines = [
        "# FragFuse Optimization Report",
        "",
        f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Artifacts",
        f"- Results directory: {results_dir}",
        f"- Fusion instructions: results/{results_name}",
        "- Optimization log: results/optimization_log.txt",
        "- Population history: results/population_history.json",
        "",
        "## Summary",
    ]

    if os.path.exists(results_file):
        try:
            analysis = analyze_optimization_results(results_file)
            score_dist = analysis.get("score_distribution", {})
            report_lines.extend(
                [
                    f"- Generations: {analysis.get('total_generations', 'N/A')}",
                    f"- Best fusion instructions: {analysis.get('num_best_individuals', 'N/A')}",
                    f"- Best score: {score_dist.get('max', 0.0):.3f}",
                ]
            )
        except Exception as e:
            report_lines.append(f"- Summary unavailable: {e}")
    else:
        report_lines.append("- Results file not found yet.")

    report_lines.extend(
        [
            "",
            "## Commands",
            "",
            "```bash",
            "python run_optimization.py",
            "python -c \"from utils import print_optimization_summary; print_optimization_summary('results/best_fusion_instructions.json')\"",
            "python -c \"from utils import plot_optimization_progress; plot_optimization_progress('results/optimization_log.txt')\"",
            "```",
        ]
    )

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    return output_file
