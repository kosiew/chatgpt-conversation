from enum import Enum
from typing import List

from icecream import ic
from pydantic import BaseModel

from structured_output.main import completions, test_response_format


class UIType(str, Enum):
    div = "div"
    button = "button"
    header = "header"
    section = "section"
    field = "field"
    form = "form"


class Attribute(BaseModel):
    name: str
    value: str


class UI(BaseModel):
    type: UIType
    label: str
    children: List["UI"]
    attributes: List[Attribute]


UI.model_rebuild()  # This is required to enable recursive types


class Response(BaseModel):
    ui: UI


messages = [
    {
        "role": "system",
        "content": "You are a UI generator AI. Convert the user input into a UI.",
    },
    {"role": "user", "content": "Make a User Profile Form"},
]
ic("test UI generation")
test_response_format(messages, completions, Response)
