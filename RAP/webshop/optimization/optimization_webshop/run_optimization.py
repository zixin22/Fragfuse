#!/usr/bin/env python3
"""Command-line runner for FragFuse fusion-instruction optimization."""

import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import Config
from evolutionary_optimizer import EvolutionaryOptimizer
from utils import (
    generate_experiment_report,
    plot_optimization_progress,
    print_optimization_summary,
    setup_experiment_directory,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="FragFuse fusion-instruction optimization")
    parser.add_argument("--max-generations", "-g", type=int, default=None)
    parser.add_argument("--population-size", "-p", type=int, default=None)
    parser.add_argument("--experiment-name", "-n", type=str, default="default_experiment")
    parser.add_argument("--config-file", "-c", type=str, default=None)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    print("=" * 80)
    print("FragFuse fusion-instruction optimization")
    print("=" * 80)
    print(f"Experiment: {args.experiment_name}")
    print()

    best_individuals = []
    config = Config()

    try:
        if args.config_file and os.path.exists(args.config_file):
            config.load_from_file(args.config_file)
            print(f"Loaded config file: {args.config_file}")

        if args.max_generations is not None:
            config.num_generations = args.max_generations
        if args.population_size is not None:
            config.population_size = args.population_size

        if args.experiment_name != "default_experiment":
            experiment_dir = setup_experiment_directory(config.base_dir, args.experiment_name)
            config.results_dir = os.path.join(experiment_dir, "results")
        else:
            config.results_dir = config.experiment_dir

        config.best_fusion_instructions_file = os.path.join(
            config.results_dir, "best_fusion_instructions.json"
        )
        config.optimization_log_file = os.path.join(config.results_dir, "optimization_log.txt")
        config.optimization_log_full_file = os.path.join(
            config.results_dir, "optimization_log_full.txt"
        )
        config.population_history_file = os.path.join(
            config.results_dir, "population_history.json"
        )
        os.makedirs(config.results_dir, exist_ok=True)

        print("Configuration:")
        print(f"  Fusion-instruction population size: {config.population_size}")
        print(f"  Macro generations: {config.num_generations}")
        print(f"  Elite size: {config.elite_size}")
        print(f"  Scoring: score = fusion_success_score - {config.lambda_ret} * L_ret")
        print(f"  Results directory: {config.results_dir}")
        print()

        optimizer = EvolutionaryOptimizer(config)
        best_individuals = optimizer.optimize()

        if best_individuals:
            print("\nTop fusion instructions:")
            ranked = sorted(best_individuals, key=lambda x: x.score, reverse=True)[:5]
            for i, ind in enumerate(ranked, 1):
                print(f"{i}. score: {ind.score:.3f} | generation: {ind.generation}")
                print(f"   instruction: {ind.prompt}")
                print()

        if not args.no_plots:
            try:
                plot_file = os.path.join(config.results_dir, "optimization_progress.png")
                plot_optimization_progress(config.optimization_log_file, plot_file)
                print(f"Saved progress plot: {plot_file}")
            except Exception as e:
                print(f"Progress plot skipped: {e}")

        print("\n" + "=" * 80)
        print("Optimization completed")
        print("=" * 80)

    except KeyboardInterrupt:
        print("\nOptimization interrupted by user")
    except Exception as e:
        print(f"\nOptimization failed: {e}")
        import traceback

        traceback.print_exc()

    try:
        config_file = os.path.join(config.results_dir, "config_used.json")
        config.save_to_file(config_file)
        print(f"Saved config: {config_file}")

        report_file = generate_experiment_report(
            config.results_dir,
            output_file=os.path.join(config.results_dir, "experiment_report.md"),
        )
        print(f"Saved report: {report_file}")

        if os.path.exists(config.best_fusion_instructions_file):
            print_optimization_summary(config.best_fusion_instructions_file)
    except Exception as e:
        print(f"Final artifact generation failed: {e}")

    return 0 if best_individuals else 1


def test_basic_functionality() -> bool:
    print("Running basic FragFuse optimization smoke test...")

    try:
        config = Config()
        print(f"Config loaded: {config}")

        from evaluator import Evaluator
        from population import Population
        from proposer import Proposer

        population = Population(config)
        evaluator = Evaluator(config)
        population.initialize_from_seeds(evaluator=evaluator)
        print(f"Population initialized: {population}")

        proposer = Proposer(config)
        candidates = proposer.generate_candidates([ind.prompt for ind in population.members[:3]])
        print(f"Generated candidates: {len(candidates)}")

        if candidates:
            scores, fusion_scores, _ = evaluator.evaluate_population(
                candidates[:3], memory_examples=[]
            )
            print(f"Evaluated candidates: {len(scores)}")
            print(f"Fusion scores: {fusion_scores}")

        print("Smoke test passed")
        return True
    except Exception as e:
        print(f"Smoke test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def show_help() -> None:
    print(
        """
FragFuse fusion-instruction optimization

Usage:
    python run_optimization.py
    python run_optimization.py --max-generations 100 -n my_experiment
    python run_optimization.py --test

Options:
    --max-generations, -g    Number of macro generations
    --population-size, -p    Fusion-instruction population size
    --experiment-name, -n    Experiment directory name
    --config-file, -c        JSON config file
    --no-plots               Skip progress plot generation
"""
    )


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ["--help", "-h", "help"]:
        show_help()
        sys.exit(0)
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        sys.exit(0 if test_basic_functionality() else 1)
    sys.exit(main())
