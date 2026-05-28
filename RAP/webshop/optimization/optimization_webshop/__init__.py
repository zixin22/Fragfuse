"""FragFuse evolutionary optimization package."""

__version__ = "1.0.0"
__author__ = "FragFuse Authors"

from .config import Config
from .population import Population, Individual
from .proposer import Proposer
from .evaluator import Evaluator
from .evolutionary_optimizer import EvolutionaryOptimizer

__all__ = [
    'Config',
    'Population',
    'Individual',
    'Proposer',
    'Evaluator',
    'EvolutionaryOptimizer'
]
