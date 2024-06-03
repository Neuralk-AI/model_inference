"""
Base class for LLMs.

Alexandre Pasquiou - April 2024
"""

import os

# import asyncio
import requests

import numpy as np
from typing import List, Any, Dict, Optional
from sentence_transformers import util

from model_inference.embeddings import BaseEmbeddingModel


class CustomEmbeddingModel(BaseEmbeddingModel):

    def __init__(
        self,
        language: str = "en",
        API_URL: str = None,
        prompts: Optional[Dict[str, str]] = None,
        default_prompt_name: Optional[str] = None,
        headers: Dict = {
            "Authorization": os.environ["HF_TOKEN"],
            "Content-Type": "application/json",
            "accept": "application/json",
        },
    ):
        """
        Instantiate an Embedding Model.
        Args:
            - model: str
        """
        self.language = language
        self.API_URL = API_URL
        self.headers = headers
        self.prompts = prompts
        self.default_prompt = (
            prompts[default_prompt_name]
            if ((default_prompt_name is not None) and (prompts is not None))
            else None
        )

    def add_default_prompt(
        self, input, default_prompt_override: str = None, prompt_name: str = None
    ):
        """Add default prompt to input"""
        if isinstance(input, str):
            query = self._add_default_prompt(
                input,
                default_prompt_override=default_prompt_override,
                prompt_name=prompt_name,
            )
        elif isinstance(input, List):
            query = [
                self._add_default_prompt(
                    item,
                    default_prompt_override=default_prompt_override,
                    prompt_name=prompt_name,
                )
                for item in input
            ]
        else:
            raise ValueError(
                f"`input` should be in ['str', 'List']. But it is {type(input)}"
            )
        return query

    def _add_default_prompt(
        self, input, default_prompt_override: str = None, prompt_name: str = None
    ):
        """Add default prompt to input"""
        if default_prompt_override is not None:
            query = default_prompt_override + input
        elif (prompt_name is not None) and (self.prompts is not None):
            query = self.prompts[prompt_name] + input
        elif self.default_prompt is not None:
            query = self.default_prompt + input
        else:
            query = input
        return query

    def embed(
        self,
        query: str | List[str],
        prompt_name: Optional[str] = None,
        prompt: Optional[str] = None,
    ) -> Any:
        """Embed the input query"""
        url = os.path.join(self.API_URL, "embed")
        query = self.add_default_prompt(
            query, default_prompt_override=prompt, prompt_name=prompt_name
        )
        embeddings = requests.post(
            url,
            headers=self.headers,
            json={
                "inputs": query,
                "truncate": False,
                "normalize": True,
            },
        )
        return np.array(embeddings.json())

    def decode(
        self,
        ids: int | List[int],
        skip_special_tokens: bool = True,
    ) -> Any:
        """Embed the input query"""
        url = os.path.join(self.API_URL, "decode")
        output = requests.post(
            url,
            headers=self.headers,
            json={
                "ids": ids,
                "skip_special_tokens": skip_special_tokens,
            },
        )
        return output

    def embed_all(
        self,
        query: str | List[str],
        prompt_name: Optional[str] = None,
        prompt: Optional[str] = None,
    ):
        """Get all Embeddings without Pooling."""
        url = os.path.join(self.API_URL, "embed_all")
        query = self.add_default_prompt(
            query, default_prompt_override=prompt, prompt_name=prompt_name
        )
        output = requests.post(
            url,
            headers=self.headers,
            json={
                "inputs": query,
                "truncate": False,
            },
        )
        return [np.array(item) for item in output.json()]

    def embed_sparse(
        self,
        query: str | List[str],
        prompt_name: Optional[str] = None,
        prompt: Optional[str] = None,
    ):
        """Get Sparse Embeddings. Returns a 424 status code if the model is not an embedding model with SPLADE pooling."""
        url = os.path.join(self.API_URL, "embed_sparse")
        query = self.add_default_prompt(
            query, default_prompt_override=prompt, prompt_name=prompt_name
        )
        output = requests.post(
            url,
            headers=self.headers,
            json={
                "inputs": query,
                "truncate": False,
            },
        )
        return np.array(output.json())

    def health(self):
        """Health check method."""
        url = os.path.join(self.API_URL, "health")
        headers = self.headers
        headers["accept"] = "*/*"
        output = requests.get(
            url,
            headers=self.headers,
        )
        return output.json()

    def info(self):
        """Text Embeddings Inference endpoint info."""
        url = os.path.join(self.API_URL, "info")
        output = requests.get(
            url,
            headers=self.headers,
        )
        return output.json()

    def metrics(self):
        """Prometheus metrics scrape endpoint."""
        url = os.path.join(self.API_URL, "metrics")
        headers = self.headers
        headers["accept"] = "text/plain"
        output = requests.get(
            url,
            headers=self.headers,
        )
        return output

    def predict(self, query: str | List[str]):
        """Get Predictions. Returns a 424 status code if the model is not a Sequence Classification model."""
        url = os.path.join(self.API_URL, "predict")
        output = requests.post(
            url,
            headers=self.headers,
            json={
                "inputs": query,
                "raw_scores": False,
                "truncate": False,
            },
        )
        return output.json()

    def rerank(
        self,
        query: str | List[str],
        texts: List[str],
        prompt_name: Optional[str] = None,
        prompt: Optional[str] = None,
    ):
        """Get Ranks. Returns a 424 status code if the model is not a Sequence Classification model with."""
        url = os.path.join(self.API_URL, "rerank")
        query = self.add_default_prompt(
            query, default_prompt_override=prompt, prompt_name=prompt_name
        )
        output = requests.post(
            url,
            headers=self.headers,
            json={
                "query": query,
                "raw_scores": False,
                "return_text": False,
                "texts": texts,
                "truncate": False,
            },
        )
        return output.json()

    def tokenize(self, query: str | List[str], add_special_tokens: bool = True):
        """Tokenize inputs."""
        url = os.path.join(self.API_URL, "tokenize")
        output = requests.post(
            url,
            headers=self.headers,
            json={
                "inputs": query,
                "add_special_tokens": add_special_tokens,
            },
        )
        return output.json()

    def compute_score(self, vector1: np.array, vector2: np.array):
        """Compute the distance between pairs of vectors"""
        score = util.dot_score(vector1, vector2)
        return score
