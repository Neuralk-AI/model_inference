# Imports
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
