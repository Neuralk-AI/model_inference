"""
Base class for LLMs.

Alexandre Pasquiou - November 2023
"""

from typing import Dict, List, Any

from model_inference.schemas import function_input_format, function_calling_format


class BaseLLM:
    function_calling_format: str = function_calling_format
    function_input_format: str = function_input_format

    def __init__(self, model: str = None, language: str = "en"):
        """
        Instantiate a LLM.
        Args:
            - model: str
        """
        self.model = model

    def format_prompt(
        self,
        system_prompt: str,
        user_prompt: str,
        functions: List[Dict] = None,
        role: str = None,
        history: List[Dict] = None,
    ) -> Any:
        """Format the prompt to send to a custom model / API"""
        raise NotImplementedError("Do your job.")

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        functions: List[Dict] = None,
        role: str = None,
        history: List[Dict] = None,
    ) -> str:
        """Send and receive a response from a LLM"""
        raise NotImplementedError("Do your job.")
