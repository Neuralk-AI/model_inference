"""
Base class for LLMs.

Alexandre Pasquiou - April 2024
"""

import numpy as np
from typing import List, Any, Dict, Optional
from sentence_transformers import SentenceTransformer, util

from model_inference.embeddings import BaseEmbeddingModel


models = [
    "all-MiniLM-L6-v2",
    "all-mpnet-base-v2",
    "gtr-t5-xxl",
    "gtr-t5-xl",
    "sentence-t5-xxl",
    "gtr-t5-large",
    "all-mpnet-base-v1",
    "multi-qa-mpnet-base-dot-v1",
    "multi-qa-mpnet-base-cos-v1",
    "all-roberta-large-v1",
    "sentence-t5-xl",
    "all-distilroberta-v1",
    "all-MiniLM-L12-v1",
    "all-MiniLM-L12-v2",
    "multi-qa-distilbert-dot-v1",
    "multi-qa-distilbert-cos-v1",
    "gtr-t5-base",
    "sentence-t5-large",
    "all-MiniLM-L6-v2",
    "multi-qa-MiniLM-L6-cos-v1",
    "all-MiniLM-L6-v1",
    "paraphrase-mpnet-base-v2",
    "msmarco-bert-base-dot-v5",
    "multi-qa-MiniLM-L6-dot-v1",
    "sentence-t5-base",
    "msmarco-distilbert-base-tas-b",
    "msmarco-distilbert-dot-v5",
    "paraphrase-distilroberta-base-v2",
    "paraphrase-MiniLM-L12-v2",
    "paraphrase-multilingual-mpnet-base-v2",
    "paraphrase-TinyBERT-L6-v2",
    "paraphrase-MiniLM-L6-v2",
    "paraphrase-albert-small-v2",
    "paraphrase-multilingual-MiniLM-L12-v2",
    "paraphrase-MiniLM-L3-v2",
    "distiluse-base-multilingual-cased-v1",
    "distiluse-base-multilingual-cased-v2",
    "average_word_embeddings_komninos",
    "average_word_embeddings_glove.6B.300d",
    "OrdalieTech/Solon-embeddings-base-0.1",
    "OrdalieTech/Solon-embeddings-large-0.1",
]


class SentenceTransformerEmbeddingModel(BaseEmbeddingModel):

    def __init__(
        self,
        model: str = "OrdalieTech/Solon-embeddings-large-0.1",
        language: str = "en",
        prompts: Optional[Dict[str, str]] = None,
        default_prompt_name: Optional[str] = None,
    ):  # multi-qa-mpnet-base-dot-v1
        """
        Instantiate an Embedding Model.
        Args:
            - model: str
        """
        self.model_name = model
        self.language = language
        self.model = SentenceTransformer(
            model_name_or_path=model,
            prompts=prompts,
            default_prompt_name=default_prompt_name,
        )

    def embed(
        self,
        query: str | List[str],
        prompt_name: Optional[str] = None,
        prompt: Optional[str] = None,
        batch_size: int = 32,
        show_progress_bar: Optional[bool] = None,
    ) -> Any:
        """Embed the input query"""
        embeddings = self.model.encode(
            sentences=query,
            prompt_name=prompt_name,
            prompt=prompt,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
        )
        return embeddings

    def compute_score(self, vector1: np.array, vector2: np.array):
        """Compute the distance between pairs of vectors"""
        score = util.dot_score(vector1, vector2)
        return score
