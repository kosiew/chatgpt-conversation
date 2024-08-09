from icecream import ic
from pydantic import BaseModel

from structured_output.structured_output import completions, test_response_format


class Step(BaseModel):
    explanation: str
    output: str


class LogicResponse(BaseModel):
    steps: list[Step]
    final_answer: str


ic("test Chain of Thought")
chain_of_thought_question = input("Enter a logical question: ")
messages = [
    {"role": "system", "content": "You are a helpful math tutor."},
    {"role": "user", "content": chain_of_thought_question},
]
response = test_response_format(messages, completions, LogicResponse)

if response.steps:  # type: ignore
    steps = response.steps  # type: ignore
    for step in steps:
        ic(step.explanation)
        ic(step.output)
    ic(response.final_answer)  # type: ignore
