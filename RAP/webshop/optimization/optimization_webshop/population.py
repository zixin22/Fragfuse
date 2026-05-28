"""Population management for FragFuse candidate instructions/templates."""

import os
import json
import random
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass


@dataclass
class Individual:
    """Candidate item in either the fusion-instruction or marker-template population."""
    prompt: str
    score: float = 0.0
    fusion_success_score: float = 0.0
    retrieval_loss: float = 0.0
    fusion_loss: float = 0.0
    objective_loss: float = 0.0
    generation: int = 0
    parent_ids: List[int] = None
    interaction_history: List[Dict[str, str]] = None

    def __post_init__(self):
        if self.parent_ids is None:
            self.parent_ids = []
        if self.interaction_history is None:
            self.interaction_history = []

    def to_dict(self) -> Dict[str, Any]:
        """"""
        return {
            'prompt': self.prompt,
            'score': self.score,
            'fusion_success_score': self.fusion_success_score,
            'retrieval_loss': self.retrieval_loss,
            'fusion_loss': self.fusion_loss,
            'objective_loss': self.objective_loss,
            'generation': self.generation,
            'parent_ids': self.parent_ids,
            'interaction_history': self.interaction_history
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Individual':
        """"""
        return cls(
            prompt=data['prompt'],
            score=data.get('score', 0.0),
            fusion_success_score=data.get('fusion_success_score', 0.0),
            retrieval_loss=data.get('retrieval_loss', 0.0),
            fusion_loss=data.get('fusion_loss', 0.0),
            objective_loss=data.get('objective_loss', 0.0),
            generation=data.get('generation', 0),
            parent_ids=data.get('parent_ids', []),
            interaction_history=data.get('interaction_history', [])
        )


class Population:
    """Population of fusion instructions or carrier-query marker templates."""

    def __init__(self, config, template_kind: str = "fusion_instruction"):
        self.config = config
        self.template_kind = template_kind
        self.size = config.population_size
        self.members: List[Individual] = []
        self.generation = 0
        self.best_individual: Optional[Individual] = None
        self.history: List[List[Individual]] = []

    @staticmethod
    def _average_history_field(interaction_history: List[Dict[str, Any]], field: str) -> float:
        values = [
            float(item[field])
            for item in interaction_history
            if isinstance(item, dict) and field in item
        ]
        return sum(values) / len(values) if values else 0.0

    def initialize_from_seeds(self, evaluator=None, fusion_instruction_file: str = None) -> None:
        """Initialize the fusion-instruction population from seed instructions."""
        if fusion_instruction_file is None:
            fusion_instruction_file = os.path.join(
                self.config.base_dir, 'data_webshop', 'fusion_instruction_seeds.txt'
            )

        if evaluator is None:
            raise ValueError("Evaluator is required to initialize fusion instructions.")

        if not os.path.exists(fusion_instruction_file):
            raise FileNotFoundError(f"Fusion-instruction seed file not found: {fusion_instruction_file}")

        fusion_instruction_templates = []
        with open(fusion_instruction_file, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            template_blocks = [p.strip() for p in content.split('\n\n') if p.strip()]
            for template_block in template_blocks:
                lines = [line.strip() for line in template_block.split('\n') if line.strip()]
                if lines:
                    template = '\n'.join(lines)
                    if template.startswith('f"') and template.endswith('"'):
                        template = template[2:-1]
                    fusion_instruction_templates.append(template)

        print(f"Loaded {len(fusion_instruction_templates)} fusion-instruction seeds")

        training_pairs = evaluator._train_pairs
        if not training_pairs:
            evaluator._load_and_split_dataset()
            training_pairs = evaluator._train_pairs

        if not training_pairs:
            raise ValueError("No training pairs available for fusion-instruction evaluation.")

        print(f"Training pairs: {len(training_pairs)}")

        template_scores = []
        for template_idx, template in enumerate(fusion_instruction_templates):
            print(f"Evaluating fusion-instruction seed {template_idx + 1}/{len(fusion_instruction_templates)}...")

            avg_score, interaction_history = evaluator.evaluate_goal_achievement(template, [])

            template_scores.append({
                'template': template,
                'avg_score': avg_score,
                'template_idx': template_idx,
                'interaction_history': interaction_history
            })

            print(f"  Seed {template_idx + 1}: avg objective score {avg_score:.4f}")

        template_scores.sort(key=lambda x: x['avg_score'], reverse=True)
        elite_n = min(self.config.elite_size, len(template_scores))
        elite_templates = template_scores[:elite_n]

        print("\n=== Elite fusion instructions ===")
        for i, elite in enumerate(elite_templates, 1):
            print(f"Elite {i}: score {elite['avg_score']:.4f}")
            print(f"  {elite['template'][:80]}...")

        self.members = []
        for elite in elite_templates:
            individual = Individual(
                prompt=elite['template'],
                score=elite['avg_score'],
                fusion_success_score=self._average_history_field(
                    elite['interaction_history'], 'fusion_success_score'
                ),
                retrieval_loss=self._average_history_field(
                    elite['interaction_history'], 'retrieval_loss'
                ),
                fusion_loss=self._average_history_field(
                    elite['interaction_history'], 'fusion_loss'
                ),
                objective_loss=self._average_history_field(
                    elite['interaction_history'], 'objective_loss'
                ),
                generation=0,
                parent_ids=[elite['template_idx']],
                interaction_history=elite['interaction_history']
            )
            self.members.append(individual)

        print(f"\nInitial fusion-instruction population size: {len(self.members)}")
        self._update_best_individual()

    def initialize_carrier_query_templates_from_file(
        self,
        evaluator,
        reference_fusion_instruction: str,
        carrier_query_template_file: str = None,
    ) -> None:
        """
        Load carrier-query template seed lines and score each template with a fixed
        fusion instruction.
        """
        if carrier_query_template_file is None:
            carrier_query_template_file = os.path.join(
                self.config.base_dir, "data_webshop", "carrier_query_template_seed.txt"
            )
        if not os.path.exists(carrier_query_template_file):
            raise FileNotFoundError(
                f"Carrier-query template seed file not found: {carrier_query_template_file}"
            )

        from carrier_query_template_utils import load_carrier_query_template_lines

        templates_all = load_carrier_query_template_lines(carrier_query_template_file)
        if not templates_all:
            raise ValueError(
                f"No carrier-query templates parsed from {carrier_query_template_file}"
            )
        # Carrier-query template initialization uses only the first seed line.
        # Additional lines can be rotated manually across runs.
        templates = [templates_all[0]]

        print(
            f"Loaded first carrier-query template seed from {carrier_query_template_file} "
            f"(1 of {len(templates_all)} lines)"
        )

        training_pairs = evaluator._train_pairs
        if not training_pairs:
            evaluator._load_and_split_dataset()
            training_pairs = evaluator._train_pairs
        if not training_pairs:
            raise ValueError("No training pairs for carrier-query template evaluation")

        template_scores = []
        for template_idx, template in enumerate(templates):
            print(f"Evaluating carrier-query template {template_idx + 1}/{len(templates)}...")
            avg_score, interaction_history = evaluator.evaluate_goal_achievement(
                reference_fusion_instruction, [], carrier_query_template=template
            )
            template_scores.append(
                {
                    "template": template,
                    "avg_score": avg_score,
                    "template_idx": template_idx,
                    "interaction_history": interaction_history,
                }
            )
            print(f"  Template {template_idx + 1}: avg score {avg_score:.4f}")

        template_scores.sort(key=lambda x: x["avg_score"], reverse=True)
        elite_n = min(self.config.elite_size, len(template_scores))
        elite_templates = template_scores[:elite_n]

        print("\n=== Elite carrier-query templates ===")
        for i, elite in enumerate(elite_templates, 1):
            print(f"Elite {i}: score {elite['avg_score']:.4f}")
            print(f"  {elite['template'][:80]}...")

        self.members = []
        for elite in elite_templates:
            self.members.append(
                Individual(
                    prompt=elite["template"],
                    score=elite["avg_score"],
                    fusion_success_score=self._average_history_field(
                        elite["interaction_history"], "fusion_success_score"
                    ),
                    retrieval_loss=self._average_history_field(
                        elite["interaction_history"], "retrieval_loss"
                    ),
                    fusion_loss=self._average_history_field(
                        elite["interaction_history"], "fusion_loss"
                    ),
                    objective_loss=self._average_history_field(
                        elite["interaction_history"], "objective_loss"
                    ),
                    generation=0,
                    parent_ids=[elite["template_idx"]],
                    interaction_history=elite["interaction_history"],
                )
            )

        print(f"\nInitial attack population size: {len(self.members)}")
        self._update_best_individual()

    def _simple_mutate(self, prompt: str) -> str:
        """Apply a lightweight mutation to fusion instructions only."""
        if self.template_kind == "carrier_query_template":
            return prompt
        # Preserve prompts that already contain a dataset mask marker.
        if '[MASK]' in prompt or '[mask]' in prompt:
            return prompt

        mutations = [
            lambda x: x.replace("Extract", "Find"),
            lambda x: x.replace("Output", "Return"),
            lambda x: x.replace("Locate", "Identify"),
            lambda x: x.replace("<product>", "[product]"),
            lambda x: x.replace("Buy", "Purchase"),
        ]

        mutation = random.choice(mutations)
        return mutation(prompt)

    def add_candidates(self, new_candidates: List[str], scores: List[float],
                      fusion_success_scores: List[float], parent_ids: List[List[int]] = None,
                      interaction_histories: List[List[Dict[str, str]]] = None,
                      generation: int = None,
                      loss_metrics: List[Dict[str, float]] = None) -> None:
        """Add evaluated candidates to the population."""
        if len(new_candidates) != len(scores):
            raise ValueError("Candidate and score lengths do not match")

        if parent_ids is None:
            parent_ids = [[] for _ in range(len(new_candidates))]
        if interaction_histories is None:
            interaction_histories = [[] for _ in range(len(new_candidates))]
        if loss_metrics is None:
            loss_metrics = [{} for _ in range(len(new_candidates))]

        new_individuals = []
        for prompt, score, fusion_score, parents, interactions, metrics in zip(
            new_candidates, scores, fusion_success_scores, parent_ids, interaction_histories, loss_metrics
        ):
            individual = Individual(
                prompt=prompt,
                score=score,
                fusion_success_score=fusion_score,
                retrieval_loss=metrics.get("retrieval_loss", 0.0),
                fusion_loss=metrics.get("fusion_loss", -fusion_score),
                objective_loss=metrics.get("objective_loss", -score),
                generation=generation if generation is not None else self.generation,
                parent_ids=parents,
                interaction_history=interactions
            )
            new_individuals.append(individual)

        self.members.extend(new_individuals)

        self._update_best_individual()

    def select_best(self, num_select: int) -> List[Individual]:
        """Return the highest-scoring individuals."""
        sorted_members = sorted(self.members, key=lambda x: x.score, reverse=True)
        return sorted_members[:num_select]

    def get_elites(self) -> List[Individual]:
        """Return elite candidates."""
        return self.select_best(self.config.elite_size)

    def evolve_population(self) -> None:
        """Keep elites and refill the population with lightweight mutations if needed."""

        elites = self.get_elites()
        remaining_slots = self.size - len(elites)

        non_elites = [ind for ind in self.members if ind not in elites]
        if non_elites:
            selected_non_elites = sorted(non_elites, key=lambda x: x.score, reverse=True)[:remaining_slots]
        else:
            selected_non_elites = []

        self.members = elites + selected_non_elites

        while len(self.members) < self.size and elites:
            elite = random.choice(elites)
            mutated_prompt = self._simple_mutate(elite.prompt)
            mutated_individual = Individual(
                prompt=mutated_prompt,
                score=elite.score * 0.9,
                generation=self.generation,
                parent_ids=[id(self)]
            )
            self.members.append(mutated_individual)

        self.generation += 1

    def _update_best_individual(self) -> None:
        """Refresh the cached best individual."""
        if self.members:
            best = max(self.members, key=lambda x: x.score)
            if self.best_individual is None or best.score > self.best_individual.score:
                self.best_individual = best

    def get_best_individual(self) -> Optional[Individual]:
        """Return the best individual observed so far."""
        return self.best_individual

    def get_statistics(self) -> Dict[str, float]:
        """Return aggregate score statistics for the population."""
        if not self.members:
            return {}

        scores = [ind.score for ind in self.members]
        fusion_scores = [ind.fusion_success_score for ind in self.members]
        retrieval_losses = [ind.retrieval_loss for ind in self.members]
        fusion_losses = [ind.fusion_loss for ind in self.members]
        objective_losses = [ind.objective_loss for ind in self.members]

        return {
            'avg_score': sum(scores) / len(scores),
            'max_score': max(scores),
            'min_score': min(scores),
            'avg_fusion_success_score': sum(fusion_scores) / len(fusion_scores),
            'avg_retrieval_loss': sum(retrieval_losses) / len(retrieval_losses),
            'avg_fusion_loss': sum(fusion_losses) / len(fusion_losses),
            'avg_objective_loss': sum(objective_losses) / len(objective_losses),
            'diversity': self._calculate_diversity()
        }

    def _calculate_diversity(self) -> float:
        """Measure prompt-level diversity."""
        if len(self.members) <= 1:
            return 0.0

        unique_prompts = set(ind.prompt for ind in self.members)
        return len(unique_prompts) / len(self.members)

    def save_history(self, file_path: str) -> None:
        """Persist population history without bulky interaction traces."""
        history_dict = {}
        for i, generation_population in enumerate(self.history):
            cleaned_generation = []
            for ind in generation_population:
                if isinstance(ind, dict):
                    d = dict(ind)
                    d.pop("interaction_history", None)
                    cleaned_generation.append(d)
                else:
                    d = ind.to_dict()
                    d.pop("interaction_history", None)
                    cleaned_generation.append(d)
            history_dict[f"population_generation_{i}"] = cleaned_generation

        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(history_dict, f, indent=2, ensure_ascii=False)

    def __len__(self) -> int:
        """Return population size."""
        return len(self.members)

    def __str__(self) -> str:
        """Return a compact population summary."""
        stats = self.get_statistics()
        return (
            f"Population(size={len(self)}, generation={self.generation}, "
            f"avg_score={stats.get('avg_score', 0):.3f}, "
            f"max_score={stats.get('max_score', 0):.3f}, "
            f"diversity={stats.get('diversity', 0):.3f})"
        )
