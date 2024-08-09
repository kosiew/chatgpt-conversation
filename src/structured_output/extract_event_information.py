from icecream import ic
from pydantic import BaseModel

from structured_output.structured_output import completions, test_response_format


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
