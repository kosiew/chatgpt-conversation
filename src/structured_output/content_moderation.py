from enum import Enum
from typing import Optional

from icecream import ic
from pydantic import BaseModel

from structured_output.main import completions, test_response_format


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
