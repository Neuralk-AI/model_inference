"""
A high-level class implementing a LLM for text generation.

Alexandre Pasquiou - November 2023
"""

import os
import json
import openai
import asyncio
from typing import List, Dict
from openai import AzureOpenAI, AsyncAzureOpenAI
from langchain_openai import AzureChatOpenAI

from tracking.progress import console
from model_inference.text_generation import BaseLLM

# "gpt-3.5-turbo-0125",
# "gpt-3.5-turbo",
# "gpt-4",
# "gpt-4-1106-preview",

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


class OpenAILLM(BaseLLM):
    def __init__(
        self,
        model: str = "gpt-3.5-turbo-0125",
        language: str = "en",
        api_version: str = "2024-02-01",
        async_=False,
        **kwargs,
    ):
        """Instantiate an OpenAI LLM.
        Args:
            - model: str
        """
        super().__init__()
        self.self_hosted = False
        self.language = language
        self.async_ = async_
        try:
            self.model_name = model
            # self.client = OpenAI(**kwargs)
            # self.llm = ChatOpenAI(model=model, temperature=0)
            if async_:
                self.client = AsyncAzureOpenAI(
                    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                    api_version=api_version,
                    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                )
            else:
                self.client = AzureOpenAI(
                    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                    api_version=api_version,
                    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                )
                self.llm = AzureChatOpenAI(
                    openai_api_version=api_version,
                    azure_deployment=model,
                )

        except NotImplementedError as ex:
            console.log(
                f"""{model} is not recognized. 
                Verify that you deployed the resources.
                Verify that you deployed the model and use the deployment name as the variable `model` here.
                Verify that your keys and endpoints are correctly registered and used.
                """
            )

    def format_prompt(
        self,
        system_prompt: str = "",
        user_prompt: str = "",
        functions: List[Dict] = None,
        role: str = "user",
        history: List[Dict] = [],
    ) -> List[Dict]:
        """For OpenAI, we want something like:
         messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"}
        ]
        Usage:
        ```python
        prompt = self.format_prompt(
            system_prompt,
            user_prompt=user_prompt,
            functions=functions,
            role=role,
            history=history,
        )```
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
        if user_prompt != "":
            messages.append({"role": role, "content": user_prompt})

        return messages

    def generate(
        self,
        messages: List[Dict[str, str]] = "",
        temperature=1,
        max_tokens=2400,
        top_p=0.95,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None,
    ) -> str:
        """
        Args:
            - messages: List[Dict[str, str]]
        Returns:
            - formatted_response: str
        """
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            stop=stop,
        )
        formatted_response = response.choices[0].message.content
        return formatted_response

    async def _agenerate(
        self,
        messages: List[Dict[str, str]] = "",
        temperature=1,
        max_tokens=2400,
        top_p=0.95,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None,
    ) -> str:
        """
        Args:
            - messages: List[Dict[str, str]]
        Returns:
            - formatted_response: str
        """
        response = await self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            stop=stop,
        )
        formatted_response = response.choices[0].message.content
        return formatted_response

    async def get_batch_responses(self, messages_list) -> List[str]:
        """
        Args:
            - messages_list: List[List[Dict[str, str]]]
        Returns:
            - batch_responses: List[str]
        """
        batch_responses = await asyncio.gather(
            *[self._agenerate(message) for message in messages_list]
        )
        return batch_responses
