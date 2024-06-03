"""
A high-level class implementing a LLM for text generation.

Alexandre Pasquiou - May 2024
"""

import json
from typing import List, Dict

from tracking.progress import console
from model_inference.text_generation import BaseLLM
from model_inference.tokenizer import Llama3Formatter, MistralFormatter, Message, Dialog

import json
import boto3


aws_models = [
    "meta.llama2-13b-chat-v1",
    "mistral.mistral-large-2402-v1:0",
    "mistral.mixtral-8x7b-instruct-v0:1",
    "mistral.mistral-7b-instruct-v0:2",
    "meta.llama3-70b-instruct-v1:0",
    "meta.llama3-8b-instruct-v1:0",
]


def load_formatter(model_id: str):
    """Load the formatter given model id."""
    if "llama3" in model_id:
        formatter = Llama3Formatter
    elif "mistral" in model_id:
        formatter = MistralFormatter
    else:
        formatter = None
    return formatter


class AWSLLM(BaseLLM):
    def __init__(
        self,
        model: str,
        language: str = "en",
        async_: bool = False,
        max_gen_len: int = 128,
        temperature: float = 0.1,
        top_p: float = 0.9,
        **kwargs,
    ):
        """Instantiate an AWS LLM endpoint.
        Args:
            - model: str
        """
        super().__init__()
        self.self_hosted = False
        self.language = language
        self.async_ = async_
        self.client = boto3.client(service_name="bedrock-runtime")
        self.accept = "application/json"
        self.content_type = "application/json"
        self.model_id = model
        self.max_gen_len = max_gen_len
        self.temperature = temperature
        self.top_p = top_p
        self.chatformatter = load_formatter(model)

    def detail_response(response):
        """Parse details from response."""
        print(f"Generated Text: {response['generation']}")
        print(f"Prompt Token count:  {response['prompt_token_count']}")
        print(f"Generation Token count:  {response['generation_token_count']}")
        print(f"Stop reason:  {response['stop_reason']}")
        return (
            response["generation"],
            response["prompt_token_count"],
            response["generation_token_count"],
            response["stop_reason"],
        )

    def generate(
        self,
        messages: List[Dict[str, str]] | List[Message] | Dialog,
    ) -> str:
        """
        Args:
            - messages: List[Dict[str, str]]
        Returns:
            - formatted_response: str
        """
        prompt = self.chatformatter.encode_dialog_prompt(messages)
        body = json.dumps(
            {
                "prompt": prompt,
                "max_gen_len": self.max_gen_len,
                "temperature": self.temperature,
                "top_p": self.top_p,
            }
        )

        response = self.client.invoke_model(
            body=body,
            modelId=self.model_id,
            accept=self.accept,
            contentType=self.content_type,
        )

        response_body = json.loads(response.get("body").read())

        return response_body
