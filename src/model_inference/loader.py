# Imports
import pickle
import numpy as np
from typing import List
from model_inference.text_generation import *

# Variables


# Functions
def _load_openai_model(name, language="en"):
    """Load model from OpenAI API."""
    model = OpenAILLM(model=name, language=language)
    return model


def _load_mistralai_model(name, language="en"):
    """Load model from MistralAI API."""
    model = MistralAILLM(model=name, language=language)
    return model


def _load_anthropic_model(name, language="en"):
    """Load model from Anthropic API."""
    model = AnthropicLLM(model=name, language=language)
    return model


def _load_cohere_model(name, language="en"):
    """Load model from Cohere API."""
    model = CohereLLM(model=name, language=language)
    return model


def _load_hf_model(name, language="en"):
    """Load model from HuggingFace API."""
    model = HuggingFaceLLM(model=name, language=language)
    return model


def _load_custom_model(name, language="en"):
    """Load model from custom API."""
    model = CustomLLM(model=name, language=language)
    return model


def load_model(name, language="en"):
    """Load model from name.
    Args:
        - name: str
    Returns:
        - model: Model
    """
    model = None
    if name in []:
        model = _load_openai_model(name, language=language)
    elif name in []:
        model = _load_mistralai_model(name, language=language)
    elif name in []:
        model = _load_anthropic_model(name, language=language)
    elif name in []:
        model = _load_cohere_model(name, language=language)
    elif name in []:
        model = _load_hf_model(name, language=language)
    else:
        model = _load_custom_model(name, language=language)

    return model


def load_embeddings(
    sentences: List[str], embeddings: np.array, path: str = "embeddings.pkl"
) -> None:
    """ """
    # Store sentences & embeddings on disc
    with open(path, "wb") as fOut:
        pickle.dump(
            {"sentences": sentences, "embeddings": embeddings},
            fOut,
            protocol=pickle.HIGHEST_PROTOCOL,
        )


def load_embeddings(path: str = "embeddings.pkl") -> tuple:
    """
    Args:
        - path: str (path to where the data is saved)
    Returns:
        - stored_sentences: List[str]
        - stored_embeddings: np.array
    """
    # Load sentences & embeddings from disc
    with open(path, "rb") as fIn:
        stored_data = pickle.load(fIn)
        stored_sentences = stored_data["sentences"]
        stored_embeddings = stored_data["embeddings"]
    return stored_sentences, stored_embeddings
