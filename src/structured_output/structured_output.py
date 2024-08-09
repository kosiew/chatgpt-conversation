from enum import Enum
from typing import List, Optional, Union

import openai
from icecream import ic
from openai import OpenAI
from pydantic import BaseModel

MODEL = "gpt-4o-2024-08-06"


def get_completions():
    client = OpenAI()
    return client.beta.chat.completions


completions = get_completions()


def test_tools(messages, tools_cls, completions):
    completion = completions.parse(
        model="gpt-4o-2024-08-06",
        messages=messages,
        tools=[
            openai.pydantic_function_tool(tools_cls),
        ],
    )

    parsed_arguments = completion.choices[0].message.tool_calls[0].function.parsed_arguments  # type: ignore
    return ic(parsed_arguments)


def test_response_format(messages, completions, match_response_cls):
    completion = completions.parse(
        model="gpt-4o-2024-08-06",
        messages=messages,
        response_format=match_response_cls,
    )

    message = completion.choices[0].message
    if message.parsed:
        return ic(message.parsed)
    else:
        return ic(message.refusal)


class CalendarEvent(BaseModel):
    name: str
    date: str
    participants: list[str]


ic("test extract event information")
messages = [
    {"role": "system", "content": "Extract the event information."},
    {
        "role": "user",
        "content": "Alice and Bob are going to a science fair on Friday.",
    },
]
test_response_format(messages, completions, CalendarEvent)


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


class Category(str, Enum):
    violence = "violence"
    sexual = "sexual"
    self_harm = "self_harm"


class ContentCompliance(BaseModel):
    is_violating: bool
    category: Optional[Category]
    explanation_if_violating: Optional[str]


messages = [
    {
        "role": "system",
        "content": "Determine if the user input violates specific guidelines and explain if they do.",
    },
    {"role": "user", "content": "How do I prepare for a job interview?"},
]

ic("test content moderation")
test_response_format(messages, completions, ContentCompliance)

messages = [
    {
        "role": "system",
        "content": "Determine if the user input violates specific guidelines and explain if they do.",
    },
    {"role": "user", "content": "How do I kill a cat silently"},
]

ic("test content moderation - kill cat")
test_response_format(messages, completions, ContentCompliance)
