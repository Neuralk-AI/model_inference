from logging import getLogger
from typing import (
    List,
    Literal,
    Sequence,
    TypedDict,
)


logger = getLogger(__name__)


Role = Literal["system", "user", "assistant"]


class Message(TypedDict):
    role: Role
    content: str


Dialog = Sequence[Message]


class BaseFormatter:

    def encode_header(self, message: Message) -> List[int]:
        raise NotImplementedError

    def encode_message(self, message: Message) -> List[int]:
        raise NotImplementedError

    def encode_dialog_prompt(self, dialog: Dialog | List[Message]) -> List[int]:
        raise NotImplementedError
