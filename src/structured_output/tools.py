from enum import Enum
from typing import Union

import openai
from icecream import ic
from openai import OpenAI
from pydantic import BaseModel

from structured_output.structured_output import MODEL, get_completions


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


messages = [
    {
        "role": "system",
        "content": "You are a helpful assistant. The current date is August 6, 2024. You help users query for the data they are looking for by calling the query function.",
    },
    {
        "role": "user",
        "content": "look up all my orders in may of last year that were fulfilled but not delivered on time",
    },
]
completions = get_completions()
test_tools(messages, Query, completions)
