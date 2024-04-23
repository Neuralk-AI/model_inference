"""
A high-level class implementing a LLM for text generation.

Alexandre Pasquiou - November 2023
"""

import os
import json
import openai

# /!\#from openai import OpenAI
from typing import List, Dict
from tracking.progress import console

from model_inference.text_generation import BaseLLM

# /!\#os.environ["OPENAI_API_KEY"]
# /!\#openai.api_version = "2023-05-15"


openai_llm_prompts = {
    "function_prompt_1": {
        "en": "You have access to the following functions: ",
        "fr": None,
    },
    "function_prompt_2": {
        "en": ". To call a function, respond - immediately and only - with a JSON object of the following format: ",
        "fr": None,
    },
}


class AnthropicLLM(BaseLLM):
    def __init__(
        self, model: str = "gpt-3.5-turbo", language: str = "en", **kwargs
    ):  # /!\#
        """Instantiate an OpenAI LLM.
        Args:
            - model: str
        """
        super().__init__()
        self.self_hosted = False
        self.language = language
        if model in ["gpt-3.5-turbo", "gpt-4", "gpt-4-1106-preview"]:  # /!\#
            # Cannot set the model = OpenAI(model) yet
            self.model = model
            # /!\#self.client = OpenAI(**kwargs)

        else:
            raise NotImplementedError(
                f"{model} is not recognized. you can choose a value in ['gpt-3.5-turbo', 'gpt-4', 'gpt-4-1106-preview']"
            )

    def format_prompt(
        self,
        system_prompt: str = "",
        user_prompt: str = "",
        functions: List[Dict] = None,
        role: str = "user",
        history: List[Dict] = [],
    ) -> List[Dict]:
        """For OpenAI, we want something like: #/!\#
         messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"}
        ]
        """
        messages = []
        function_prompt = ""
        if functions is not None:
            if not isinstance(functions, list):
                functions = [functions]
            function_prompt = openai_llm_prompts["function_prompt_1"][self.language]
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
                openai_llm_prompts["function_prompt_2"][self.language]
                + self.function_calling_format
                + ". Only use the functions if required !"
            )

        messages.append(
            {"role": "system", "content": " ".join([system_prompt, function_prompt])}
        )
        if history != []:
            messages.extend(history)
        # messages.append({"role": role, "content": user_prompt})

        return messages

    def generate(
        self,
        system_prompt: str = "",
        user_prompt: str = "",
        functions: List[Dict] = None,
        role: str = None,
        history: List[Dict] = [],
    ) -> str:
        """
        Args:
            - system_prompt: str
            - user_prompt: str
            - functions: List[Dict]
        Returns:
            - formatted_response: str
        """
        prompt = self.format_prompt(
            system_prompt,
            user_prompt=user_prompt,
            functions=functions,
            role=role,
            history=history,
        )
        # console.log("-------------> PROMPT:", prompt)
        # Not yet possible unfortunately to add temperature parameter:
        # Only possibility is a seed using the system_fingerprint argument
        response = self.client.chat.completions.create(
            model=self.model, messages=prompt
        )

        formatted_response = response.choices[0].message.content

        return formatted_response
