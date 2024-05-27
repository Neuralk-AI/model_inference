"""
Base class for LLMs.

Alexandre Pasquiou - April 2024
"""

from typing import Dict, Optional

from model_inference.embeddings.sentencetransformer import (
    st_models,
    SentenceTransformerEmbeddingModel,
)
from model_inference.embeddings.customembedder import CustomEmbeddingModel


class AutoEmbeddingModel:

    @staticmethod
    def from_pretrained(
        model: str = None,
        language: str = "en",
        API_URL: str = None,
        prompts: Optional[Dict[str, str]] = None,
        default_prompt_name: Optional[str] = None,
        headers: Dict = {
            "Authorization": "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",
            "Content-Type": "application/json",
            "accept": "application/json",
        },
    ):
        if (API_URL is None) and (model in st_models):
            return SentenceTransformerEmbeddingModel(
                model=model,
                language=language,
                prompts=prompts,
                default_prompt_name=default_prompt_name,
            )
        elif API_URL is not None:
            return CustomEmbeddingModel(
                API_URL=API_URL,
                headers=headers,
                language=language,
                prompts=prompts,
                default_prompt_name=default_prompt_name,
            )
        else:
            raise ValueError(
                f"Invalid arguments: missing `API_URL` or `model` that should be in {st_models}"
            )
