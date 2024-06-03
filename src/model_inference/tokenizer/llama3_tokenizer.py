from logging import getLogger
from typing import (
    List,
    Literal,
    Sequence,
    TypedDict,
)
from model_inference.tokenizer import BaseFormatter

logger = getLogger(__name__)


Role = Literal["system", "user", "assistant"]


class Message(TypedDict):
    role: Role
    content: str


Dialog = Sequence[Message]


class Llama3Formatter(BaseFormatter):

    def encode_header(self, message: Message) -> List[int]:
        prompt = ""
        prompt += "<|start_header_id|>"
        prompt += message["role"]
        prompt += "<|end_header_id|>"
        prompt += "\n\n"
        return prompt

    def encode_message(self, message: Message) -> List[int]:
        prompt = self.encode_header(message)
        prompt += message["content"].strip()
        prompt += "<|eot_id|>"
        return prompt

    def encode_dialog_prompt(self, dialog: Dialog | List[Message]) -> List[int]:
        prompt = ""
        prompt += "<|begin_of_text|>"
        for message in dialog:
            prompt += self.encode_message(message)
        # Add the start of an assistant message for the model to complete.
        prompt += self.encode_header({"role": "assistant", "content": ""})
        return prompt
