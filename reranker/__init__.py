"""
Reranking module for baseline methods.

This module provides implementations of MMR (Maximum Marginal Relevance) and
DPP (Determinantal Point Process) reranking algorithms.
"""

from reranker.dpp import DPP
from reranker.mmr import MMR

__all__ = ["MMR", "DPP"]
