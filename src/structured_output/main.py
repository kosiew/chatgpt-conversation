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
