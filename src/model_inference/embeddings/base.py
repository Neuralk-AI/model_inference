"""
Base class for LLMs.

Alexandre Pasquiou - April 2024
"""

from typing import List, Any
import numpy as np


class BaseEmbeddingModel:

    def __init__(self, model: str = None, language: str = "en"):
        """
        Instantiate an Embedding Model.
        Args:
            - model: str
        """
        self.model_name = model
        self.language = language

    def embed(self, query: str | List[str]) -> Any:
        """Embed the input query"""
        raise NotImplementedError("Do your job.")

    def compute_score(self, vector1: np.array, vector2: np.array):
        """Compute the distance between pairs of vectors"""
        raise NotImplementedError("Do your job.")
