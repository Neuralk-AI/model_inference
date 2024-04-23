"""
A high-level class implementing a LLM for text generation.

Alexandre Pasquiou - November 2023
"""

import json
from typing import List, Dict
from huggingface_hub import InferenceClient
from transformers import AutoTokenizer, pipeline
from optimum.onnxruntime import ORTModelForQuestionAnswering

from tracking.progress import console

from model_inference.text_generation import BaseLLM

custom_llm_prompts = {
    "function_prompt_1": {
        "en": "You have access to the following functions: ",
        "fr": None,
    },
    "function_prompt_2": {
        "en": " To call a function, respond - immediately and only - with a JSON object of the following format: ",
        "fr": None,
    },
}


class CustomLLM(BaseLLM):
    def __init__(
        self,
        model: str = "unknown",
        self_hosted: bool = True,
        endpoint_ip: str = None,
        port: int = None,
        language: str = "en",
        **kwargs,
    ) -> None:
        """Instantiate a custom LLM.
        Args:
            - model: str
            - endpoint_ip: str | The endpoint ip at which the model is hosted
            - port: int | The endpoint port number

        Kwargs should integrate the following arguments:
            - max_new_tokens: int = 20,
            - best_of: Optional[int] = None,
            - repetition_penalty: Optional[float] = None,
            - return_full_text: bool = False,
            - seed: Optional[int] = None,
            - stop_sequences: Optional[List[str]] = None,
            - temperature: Optional[float] = None,
            - top_k: Optional[int] = None,
            - top_p: Optional[float] = None,
        """
        self.self_hosted = self_hosted
        self.model = model
        self.language = language
        # Run using an endpoint we defined ourselves, defined by the model (which is)
        self.endpoint_ip = endpoint_ip  # Eg: "3.250.186.244"
        self.port = port  # Eg: 8080
        self.metadata = kwargs  # ['key']  # If needed for access / security reasons

        self.init_inference_model()

    def init_inference_model(self) -> None:
        """Initialize the LLM model. Whether you run it locally or
        on some API.
        """
        if self.ip and self.port:
            self.client = InferenceClient(model=f"http://{self.ip}:{self.port}")
        else:
            raise NotImplementedError("Do your job.")

    def format_prompt(
        self,
        system_prompt: str = "",
        user_prompt: str = "",
        functions: List[Dict] = None,
        history: List[Dict] = None,
        role: str = None,
    ) -> str:
        """ """
        B_INST, E_INST = "[INST]", "[/INST]\n\n"
        B_SYS, E_SYS = "<SYS>", "</SYS>\n\n"

        function_prompt = ""
        if functions is not None:
            if not isinstance(functions, list):
                functions = [functions]
            B_FUNC, E_FUNC = " <FUNCTIONS>", "</FUNCTIONS>\n"
            function_prompt = (
                custom_llm_prompts["function_prompt_1"][self.language] + B_FUNC
            )
            for function in functions:
                function_dict = {
                    "name": function["name"],
                    "description": function["description"],
                    "args": [
                        {
                            "name": arg["name"],
                            "type": arg["type"],
                            "description": arg["description"],
                            "required": arg["required"],
                        }
                        for arg in function["args"]
                    ],
                }
                function_string = json.dumps(function_dict)
                function_prompt += function_string
            function_prompt += (
                E_FUNC
                + custom_llm_prompts["function_prompt_2"][self.language]
                + self.function_calling_format
                + ". Only use the functions if required !"
            )

            system_prompt = " ".join([system_prompt.strip(), function_prompt.strip()])
        # May be put the `B_INST` symbol after `E_SYS`... TODO
        prompt = f"{B_INST}{B_SYS}{system_prompt}{E_SYS}{user_prompt.strip()}{E_INST}"
        raise NotImplementedError("TODO: integrate history in prompt.")

        return prompt

    def generate(
        self,
        system_prompt: str = "",
        user_prompt: str = "",
        functions: List[Dict] = None,
        history: List[Dict] = None,
        role: str = None,
    ) -> str:
        """
        Args:
            - system_prompt: str
            - instruction: str
            - functions: List[Dict]
        """
        prompt = self.format_prompt(
            system_prompt,
            user_prompt=user_prompt,
            functions=functions,
            history=history,
            role=role,
        )

        formatted_response = self.client.text_generation(prompt=prompt, **self.metadata)

        return formatted_response
