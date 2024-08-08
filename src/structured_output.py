from enum import Enum
from typing import Union

import openai
from icecream import ic
from openai import OpenAI
from pydantic import BaseModel


class Table(str, Enum):
    orders = "orders"
    customers = "customers"
    products = "products"


class Column(str, Enum):
    id = "id"
    status = "status"
    expected_delivery_date = "expected_delivery_date"
    delivered_at = "delivered_at"
    shipped_at = "shipped_at"
    ordered_at = "ordered_at"
    canceled_at = "canceled_at"


class Operator(str, Enum):
    eq = "="
    gt = ">"
    lt = "<"
    le = "<="
    ge = ">="
    ne = "!="


class OrderBy(str, Enum):
    asc = "asc"
    desc = "desc"


class DynamicValue(BaseModel):
    column_name: str


class Condition(BaseModel):
    column: str
    operator: Operator
    value: Union[str, int, DynamicValue]


class Query(BaseModel):
    table_name: Table
    columns: list[Column]
    conditions: list[Condition]
    order_by: OrderBy


client = OpenAI()

completions = client.beta.chat.completions


def test_tools(query_cls, completions):
    completion = completions.parse(
        model="gpt-4o-2024-08-06",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant. The current date is August 6, 2024. You help users query for the data they are looking for by calling the query function.",
            },
            {
                "role": "user",
                "content": "look up all my orders in may of last year that were fulfilled but not delivered on time",
            },
        ],
        tools=[
            openai.pydantic_function_tool(query_cls),
        ],
    )

    parsed_arguments = completion.choices[0].message.tool_calls[0].function.parsed_arguments  # type: ignore
    ic(parsed_arguments)


test_tools(Query, completions)


class Step(BaseModel):
    explanation: str
    output: str


class MathResponse(BaseModel):
    steps: list[Step]
    final_answer: str


def test_response_format(completions, match_response_cls):
    completion = completions.parse(
        model="gpt-4o-2024-08-06",
        messages=[
            {"role": "system", "content": "You are a helpful math tutor."},
            {"role": "user", "content": "solve 8x + 31 = 2"},
        ],
        response_format=match_response_cls,
    )

    message = completion.choices[0].message
    if message.parsed:
        ic(message.parsed.steps)
        ic(message.parsed.final_answer)
    else:
        ic(message.refusal)


ic("New test")
test_response_format(completions, MathResponse)
