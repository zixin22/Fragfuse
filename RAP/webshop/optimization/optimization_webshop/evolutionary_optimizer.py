"""Evolutionary optimizer for FragFuse fusion instructions and carrier queries."""

import os
import time
import json
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

from config import Config
from population import Population, Individual
from proposer import Proposer
from evaluator import Evaluator
from symbol_proposer import SymbolProposer


class EvolutionaryOptimizer:
    """Alternates between fusion-instruction and carrier-query template optimization."""

    def __init__(self, config: Config):
        self.config = config
        self.population = Population(config, template_kind="fusion_instruction")
        self.carrier_population = Population(config, template_kind="carrier_query_template")
        self.proposer = Proposer(config)
        self.symbol_proposer = SymbolProposer(config)
        self.evaluator = Evaluator(config)

        self.current_generation = 0
        self.best_individual: Optional[Individual] = None
        self.start_time: Optional[float] = None

        self.optimization_log: List[Dict[str, Any]] = []
        self.optimization_log_full: List[Dict[str, Any]] = []

    def optimize(self, max_generations: Optional[int] = None) -> List[Individual]:
        """Run alternating optimization over both FragFuse design components."""
        if max_generations is None:
            max_generations = self.config.num_generations

        print("=" * 80)
        print("FragFuse optimization (fusion instruction / marker template)")
        print("=" * 80)
        print(f"Macro generations: {max_generations}")
        print(f"Population size: {self.config.population_size}")
        print()

        self.start_time = time.time()

        self._initialize_population()
        self._initialize_carrier_population()

        best_individuals: List[Individual] = []
        no_improvement_count = 0
        previous_best_joint_score = -1.0
        subgeneration_counter = 0

        for macro_generation in range(max_generations):
            print(f"\n{'=' * 60}\nMacro generation {macro_generation + 1}/{max_generations}\n{'=' * 60}")

            # ----- Fusion-instruction phase (fix carrier-query template) -----
            self.current_generation = subgeneration_counter
            fixed_carrier_template = self._current_best_carrier_query_template()
            print(f"\n--- Fusion-instruction phase {subgeneration_counter + 1} ---")
            print(f"[Fixed carrier-query template] {fixed_carrier_template}")

            candidates = self._generate_candidates()
            if not candidates:
                print("No fusion-instruction candidates generated")
            else:
                print(f"Generated {len(candidates)} fusion-instruction candidates")
                memory_examples: List = []
                total_scores, goal_scores, interaction_histories = self.evaluator.evaluate_population(
                    candidates, memory_examples=memory_examples, carrier_query_template=fixed_carrier_template
                )
                self._add_candidates_to_population(
                    candidates, total_scores, goal_scores, interaction_histories, self.population
                )
                self._select_and_update_population(self.population)

            self.population.history.append([ind.to_dict() for ind in self.population.members])
            current_best_fusion_instruction = self.population.get_best_individual()
            if current_best_fusion_instruction:
                best_individuals.append(current_best_fusion_instruction)
                print(f"Best fusion-instruction score: {current_best_fusion_instruction.score:.3f}")

            self._log_generation_info(
                subgeneration_counter,
                current_best_fusion_instruction,
                phase="fusion_instruction",
                carrier_best=self.carrier_population.get_best_individual(),
                fixed_carrier_query_template=fixed_carrier_template,
            )
            subgeneration_counter += 1

            # ----- Carrier-query template phase (fix fusion instruction) -----
            self.current_generation = subgeneration_counter
            fix_fusion_instruction = self._current_best_fusion_instruction()
            print(f"\n--- Carrier-query template phase {subgeneration_counter + 1} ---")
            fusion_preview = (
                fix_fusion_instruction[:200] + "..." if fix_fusion_instruction and len(fix_fusion_instruction) > 200 else fix_fusion_instruction
            )
            print(f"[Fixed fusion instruction] {fusion_preview}")

            carrier_candidates = self._generate_carrier_query_candidates()
            if not carrier_candidates:
                print("No carrier-query template candidates generated")
            else:
                print(f"Generated {len(carrier_candidates)} carrier-query template candidates")
                memory_examples = []
                at_total, at_goal, at_hist = self.evaluator.evaluate_carrier_query_templates(
                    fix_fusion_instruction, carrier_candidates, memory_examples=memory_examples
                )
                self._add_candidates_to_population(
                    carrier_candidates, at_total, at_goal, at_hist, self.carrier_population
                )
                self._select_and_update_population(self.carrier_population)

            self.carrier_population.history.append([ind.to_dict() for ind in self.carrier_population.members])
            current_best_carrier_template = self.carrier_population.get_best_individual()
            print(
                f"Best carrier-query template score: {current_best_carrier_template.score:.3f}"
                if current_best_carrier_template
                else "No carrier-query template selected"
            )

            self._log_generation_info(
                subgeneration_counter,
                current_best_fusion_instruction,
                phase="carrier_query_template",
                carrier_best=current_best_carrier_template,
                fixed_fusion_instruction=fix_fusion_instruction,
            )
            subgeneration_counter += 1

            # Joint score for early stopping: best fusion instruction plus best carrier query.
            joint_score = self._evaluate_joint_best()
            print(f"Joint fusion/carrier score: {joint_score:.4f}")

            if self._check_termination_conditions_joint(no_improvement_count):
                print("Stopping after repeated non-improvement")
                break

            if joint_score > previous_best_joint_score:
                previous_best_joint_score = joint_score
                no_improvement_count = 0
            else:
                no_improvement_count += 1

        self._finalize_optimization(best_individuals)
        return best_individuals

    def _current_best_carrier_query_template(self) -> str:
        ind = self.carrier_population.get_best_individual()
        if ind:
            return ind.prompt
        return self.evaluator._carrier_query_template

    def _current_best_fusion_instruction(self) -> str:
        ind = self.population.get_best_individual()
        if ind:
            return ind.prompt
        return ""

    def _evaluate_joint_best(self) -> float:
        bt = self.population.get_best_individual()
        ba = self.carrier_population.get_best_individual()
        if not bt or not ba:
            return 0.0
        score, _ = self.evaluator.evaluate_goal_achievement(
            bt.prompt, [], carrier_query_template=ba.prompt
        )
        return float(score)

    def _check_termination_conditions_joint(self, no_improvement_count: int) -> bool:
        if no_improvement_count >= self.config.no_improvement_generations:
            print(f"No improvement for {no_improvement_count} generations")
            return True
        return False

    def _initialize_carrier_population(self) -> None:
        print("Initializing carrier-query template population...")
        ref = self._current_best_fusion_instruction()
        if not ref:
            raise RuntimeError("Cannot initialize carrier-query templates without a fusion instruction")
        try:
            self.carrier_population.initialize_carrier_query_templates_from_file(
                self.evaluator, reference_fusion_instruction=ref
            )
            print(f"Carrier-query template population size: {len(self.carrier_population)}")
        except Exception as e:
            print(f"Carrier-query template initialization failed: {e}")
            raise

    def _generate_carrier_query_candidates(self) -> List[str]:
        current = [ind.prompt for ind in self.carrier_population.members]
        return self.symbol_proposer.generate_candidates(current)

    def _initialize_population(self) -> None:
        """Initialize the fusion-instruction population from seed instructions."""
        print("Initializing fusion-instruction population...")
        try:
            self.population.initialize_from_seeds(evaluator=self.evaluator)
            print(f"Fusion-instruction population size: {len(self.population)}")
        except Exception as e:
            print(f"Fusion-instruction initialization failed: {e}")
            raise

    def _generate_candidates(self) -> List[str]:
        """Generate fusion-instruction candidates from the current population."""
        current_prompts = [ind.prompt for ind in self.population.members]
        return self.proposer.generate_candidates(current_prompts)

    def _add_candidates_to_population(
        self,
        candidates: List[str],
        total_scores: List[float],
        goal_scores: List[float],
        interaction_histories: List[List[Dict[str, str]]],
        population: Optional[Population] = None,
    ) -> None:
        """Add evaluated candidates to the requested population."""
        pop = population if population is not None else self.population
        parent_ids = [[i] for i in range(len(candidates))]
        current_gen = self.current_generation
        loss_metrics = [self._average_loss_metrics(history) for history in interaction_histories]

        pop.add_candidates(
            candidates,
            total_scores,
            goal_scores,
            parent_ids,
            interaction_histories,
            generation=current_gen,
            loss_metrics=loss_metrics,
        )

    @staticmethod
    def _average_loss_metrics(interaction_history: List[Dict[str, Any]]) -> Dict[str, float]:
        metrics: Dict[str, float] = {}
        for field in ("fusion_success_score", "retrieval_loss", "fusion_loss", "objective_loss"):
            values = [
                float(item[field])
                for item in interaction_history
                if isinstance(item, dict) and field in item
            ]
            metrics[field] = sum(values) / len(values) if values else 0.0
        return metrics

    def _select_and_update_population(self, population: Optional[Population] = None) -> None:
        """Select survivors and update the requested population."""
        pop = population if population is not None else self.population
        pop.evolve_population()

    def _log_generation_info(
        self,
        generation: int,
        current_best: Optional[Individual],
        phase: str,
        carrier_best: Optional[Individual] = None,
        fixed_carrier_query_template: Optional[str] = None,
        fixed_fusion_instruction: Optional[str] = None,
    ) -> None:
        """Append compact and full generation logs."""
        def _clean_individual(ind: Optional[Individual]) -> Optional[Dict[str, Any]]:
            if not ind:
                return None
            d = ind.to_dict()
            d.pop("interaction_history", None)
            return d

        def _full_individual(ind: Optional[Individual]) -> Optional[Dict[str, Any]]:
            if not ind:
                return None
            return ind.to_dict()

        stats_fusion = self.population.get_statistics()
        stats_carrier = self.carrier_population.get_statistics()

        log_entry = {
            'generation': generation,
            'phase': phase,
            'timestamp': datetime.now().isoformat(),
            'population_size_fusion_instruction': len(self.population),
            'population_size_carrier_query_template': len(self.carrier_population),
            'statistics_fusion_instruction': stats_fusion,
            'statistics_carrier_query_template': stats_carrier,
            'best_individual': _clean_individual(current_best),
            'best_carrier_query_template_individual': _clean_individual(carrier_best),
            'fixed_carrier_query_template': fixed_carrier_query_template,
            'fixed_carrier_query_template_full': fixed_carrier_query_template,
            'fixed_fusion_instruction_full': fixed_fusion_instruction,
            'fixed_fusion_instruction_preview': (fixed_fusion_instruction[:120] + "...") if fixed_fusion_instruction and len(fixed_fusion_instruction) > 120 else fixed_fusion_instruction,
            'diversity': stats_fusion.get('diversity', 0.0),
            'elapsed_time': time.time() - self.start_time if self.start_time else 0,
        }
        log_entry_full = {
            'generation': generation,
            'phase': phase,
            'timestamp': log_entry['timestamp'],
            'population_size_fusion_instruction': len(self.population),
            'population_size_carrier_query_template': len(self.carrier_population),
            'statistics_fusion_instruction': stats_fusion,
            'statistics_carrier_query_template': stats_carrier,
            'best_individual': _full_individual(current_best),
            'best_carrier_query_template_individual': _full_individual(carrier_best),
            'fixed_carrier_query_template': fixed_carrier_query_template,
            'fixed_carrier_query_template_full': fixed_carrier_query_template,
            'fixed_fusion_instruction_full': fixed_fusion_instruction,
            'fixed_fusion_instruction_preview': log_entry['fixed_fusion_instruction_preview'],
            'diversity': stats_fusion.get('diversity', 0.0),
            'elapsed_time': log_entry['elapsed_time'],
        }

        self.optimization_log.append(log_entry)
        self.optimization_log_full.append(log_entry_full)

        self._save_optimization_log()

    def _finalize_optimization(self, best_individuals: List[Individual]) -> None:
        """Save final results and print the final summary."""
        elapsed_time = time.time() - self.start_time if self.start_time else 0

        print("\n" + "=" * 80)
        print("Optimization finished")
        print("=" * 80)
        print(f"Elapsed time: {elapsed_time:.2f}s")
        print(f"Subgenerations: {self.current_generation + 1}")
        print(f"Best fusion-instruction records: {len(best_individuals)}")

        if best_individuals:
            final_best = max(best_individuals, key=lambda x: x.score)
            print("\nFinal best fusion instruction:")
            print(f"  Prompt: {final_best.prompt}")
            print(f"  Score: {final_best.score:.3f}")
            print(f"  fusion_success_score: {final_best.fusion_success_score:.3f}")
            print(f"  retrieval_loss: {final_best.retrieval_loss:.3f}")
            print(f"  objective_loss: {final_best.objective_loss:.3f}")
            print(f"  Generation: {final_best.generation}")

        ab = self.carrier_population.get_best_individual()
        if ab:
            print("\nFinal best carrier-query template:")
            print(f"  {ab.prompt}")
            print(f"  Score: {ab.score:.3f}")

        self._save_final_results(best_individuals)

    def _save_optimization_log(self) -> None:
        """Persist compact and full optimization logs."""
        try:
            with open(self.config.optimization_log_file, 'w', encoding='utf-8') as f:
                json.dump(self.optimization_log, f, indent=2, ensure_ascii=False)
            full_file = getattr(self.config, "optimization_log_full_file", None)
            if full_file:
                with open(full_file, 'w', encoding='utf-8') as f:
                    json.dump(self.optimization_log_full, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Failed to save optimization log: {e}")

    def _save_final_results(self, best_individuals: List[Individual]) -> None:
        """Persist final optimized instructions, test evaluation, and histories."""
        try:
            final_best = max(best_individuals, key=lambda x: x.score) if best_individuals else None

            test_score = 0.0
            test_interaction_history = []
            final_best_carrier_template = self.carrier_population.get_best_individual()
            carrier_template_for_test = final_best_carrier_template.prompt if final_best_carrier_template else None

            if final_best:
                try:
                    test_score, test_interaction_history = self.evaluator.evaluate_on_test_set(
                        final_best.prompt, carrier_query_template=carrier_template_for_test
                    )
                    test_metrics = self._average_loss_metrics(test_interaction_history)
                    print(f"Test-set objective score: {test_score:.4f}")
                    print(
                        f"Test-set fusion_success_score: "
                        f"{test_metrics.get('fusion_success_score', 0.0):.4f}"
                    )

                    # optimization_log
                    if test_interaction_history:
                        test_log_entry = {
                            'generation': 'test_evaluation',
                            'timestamp': datetime.now().isoformat(),
                            'test_score': test_score,
                            'total_test_pairs': len(test_interaction_history),
                            'carrier_query_template_used': carrier_template_for_test,
                        }
                        self.optimization_log.append(test_log_entry)
                        test_log_entry_full = dict(test_log_entry)
                        test_log_entry_full['test_interactions'] = test_interaction_history
                        self.optimization_log_full.append(test_log_entry_full)

                except Exception as e:
                    print(f"Test-set evaluation failed: {e}")

            def clean_individual_dict(ind):
                d = ind.to_dict()
                d.pop('interaction_history', None)
                return d

            best_data = {
                'optimization_completed_at': datetime.now().isoformat(),
                'total_generations': self.current_generation + 1,
                'best_individuals': [clean_individual_dict(ind) for ind in best_individuals],
                'final_best': clean_individual_dict(final_best) if final_best else None,
                'final_best_carrier_query_template': clean_individual_dict(final_best_carrier_template)
                if final_best_carrier_template
                else None,
                'test_set_score': test_score,
                'config': self.config.to_dict(),
            }

            with open(self.config.best_fusion_instructions_file, 'w', encoding='utf-8') as f:
                json.dump(best_data, f, indent=2, ensure_ascii=False)

            # population_history
            if test_interaction_history:
                # population_history
                test_generation_data = {
                    f'population_test_evaluation': [{
                        'prompt': final_best.prompt if final_best else '',
                        'carrier_query_template': carrier_template_for_test or '',
                        'score': test_score,
                        'fusion_success_score': self._average_loss_metrics(test_interaction_history).get('fusion_success_score', test_score),
                        'retrieval_loss': self._average_loss_metrics(test_interaction_history).get('retrieval_loss', 0.0),
                        'fusion_loss': self._average_loss_metrics(test_interaction_history).get('fusion_loss', -test_score),
                        'objective_loss': self._average_loss_metrics(test_interaction_history).get('objective_loss', -test_score),
                        'generation': 'test_evaluation',
                        'parent_ids': [],
                        'interaction_history': test_interaction_history
                    }]
                }

                try:
                    if os.path.exists(self.config.population_history_file):
                        with open(self.config.population_history_file, 'r', encoding='utf-8') as f:
                            existing_data = json.load(f)
                    else:
                        existing_data = {}

                    existing_data.update(test_generation_data)

                    with open(self.config.population_history_file, 'w', encoding='utf-8') as f:
                        json.dump(existing_data, f, indent=2, ensure_ascii=False)
                except Exception as e:
                    print(f"Failed to append test evaluation to population history: {e}")

            self.population.save_history(self.config.population_history_file)
            carrier_history_file = os.path.join(
                os.path.dirname(self.config.population_history_file),
                "population_history_carrier_query_template.json",
            )
            self.carrier_population.save_history(carrier_history_file)

            self._save_optimization_log()

            print(f"Saved optimization artifacts under: {self.config.experiment_dir}")

        except Exception as e:
            print(f"Failed to save final results: {e}")
            import traceback
            traceback.print_exc()

    def get_optimization_summary(self) -> Dict[str, Any]:
        """Return a compact summary of the current optimization run."""
        if not self.optimization_log:
            return {}

        final_log = self.optimization_log[-1]
        best_individuals = [log['best_individual'] for log in self.optimization_log
                           if log.get('best_individual') is not None]

        pop_sz = final_log.get('population_size_fusion_instruction', final_log.get('population_size', 0))

        stats_fusion = final_log.get('statistics_fusion_instruction', final_log.get('statistics', {}))

        return {
            'total_generations': len(self.optimization_log),
            'final_population_size': pop_sz,
            'final_statistics': stats_fusion,
            'best_score_progression': [ind['score'] for ind in best_individuals] if best_individuals else [],
            'optimization_time': final_log.get('elapsed_time', 0),
        }

    def __str__(self) -> str:
        """Return a concise optimizer status string."""
        summary = self.get_optimization_summary()
        return (
            f"EvolutionaryOptimizer(generations={summary.get('total_generations', 0)}, "
            f"population_size={summary.get('final_population_size', 0)}, "
            f"optimization_time={summary.get('optimization_time', 0):.1f}s)"
        )
