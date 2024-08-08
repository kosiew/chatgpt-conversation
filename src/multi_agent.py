import json
from enum import Enum
from io import StringIO

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from icecream import ic
from IPython.display import Image
from openai import BaseModel, OpenAI, pydantic_function_tool

client = OpenAI()
MODEL = "gpt-4o-2024-08-06"

triaging_system_prompt = """You are a Triaging Agent. Your role is to assess the user's query and route it to the relevant agents. The agents available are:
- Data Processing Agent: Cleans, transforms, and aggregates data.
- Analysis Agent: Performs statistical, correlation, and regression analysis.
- Visualization Agent: Creates bar charts, line charts, and pie charts.

Use the send_query_to_agents tool to forward the user's query to the relevant agents. Also, use the speak_to_user tool to get more information from the user if needed."""

processing_system_prompt = """You are a Data Processing Agent. Your role is to clean, transform, and aggregate data using the following tools:
- CleanDataTool
- TransformDataTool
- AggregateDataTool"""

analysis_system_prompt = """You are an Analysis Agent. Your role is to perform statistical, correlation, and regression analysis using the following tools:
- StatAnalysisTool
- CorrelationAnalysisTool
- RegressionAnalysisTool"""

visualization_system_prompt = """You are a Visualization Agent. Your role is to create bar charts, line charts, and pie charts using the following tools:
- CreateBarChartTool
- CreateLineChartTool
- CreatePieChartTool """


class Agent(str, Enum):
    Data_Processing = "Data Processing Agent"
    Analysis = "Analysis Agent"
    Visualization = "Visualization Agent"


class Triage(BaseModel):
    agents: list[Agent]
    query: str

    class Config:
        extra = "forbid"  # Ensure no additional properties are allowed


class CleanDataTool(BaseModel):
    """Cleans the provided data by removing duplicates and handling missing values."""

    data: str

    class Config:
        extra = "forbid"


class TransformDataTool(BaseModel):
    """Transforms data based on specified rules."""

    data: str
    rules: str

    class Config:
        extra = "forbid"


class AggregateDataTool(BaseModel):
    """Aggregates data by specified columns and operations."""

    data: str
    group_by: list[str]
    operations: str

    class Config:
        extra = "forbid"


class StatAnalysisTool(BaseModel):
    """Peforms statistical analysis on the given dataset.
    Args:
        BaseModel (_type_): _description_
    """

    data: str

    class Config:
        extra = "forbid"


class CorrelationAnalysisTool(BaseModel):
    """Calculates correlation coefficients between variables in the dataset.

    Args:
        BaseModel (_type_): _description_
    """

    data: str
    variables: list[str]

    class Config:
        extra = "forbid"


class RegressionAnalysisTool(BaseModel):
    """Performs regression analysis on the dataset.

    Args:
        BaseModel (_type_): _description_
    """

    data: str
    dependent_var: str
    independent_vars: list[str]

    class Config:
        extra = "forbid"


class CreateBarChartTool(BaseModel):
    """Creates a bar chart from the provided data.

    Args:
        BaseModel (_type_): _description_
    """

    data: str
    x: str
    y: str

    class Config:
        extra = "forbid"


class CreateLineChartTool(BaseModel):
    """Creates a line chart from the provided data.

    Args:
        BaseModel (_type_): _description_
    """

    data: str
    x: str
    y: str

    class Config:
        extra = "forbid"


class CreatePieChartTool(BaseModel):
    """Creates a pie chart from the provided data.

    Args:
        BaseModel (_type_): _description_
    """

    data: str
    labels: str
    values: str

    class Config:
        extra = "forbid"


# Example query

user_query = """
Below is some data. I want you to first remove the duplicates then analyze the statistics of the data as well as plot a line chart.

house_size (m3), house_price ($)
90, 100
80, 90
100, 120
90, 100
"""


def clean_data(data):
    data_io = StringIO(data)
    df = pd.read_csv(data_io, sep=",")
    df_deduplicated = df.drop_duplicates()
    return df_deduplicated


def stat_analysis(data):
    data_io = StringIO(data)
    df = pd.read_csv(data_io, sep=",")
    return df.describe()


def plot_line_chart(data):
    data_io = StringIO(data)
    df = pd.read_csv(data_io, sep=",")

    x = df.iloc[:, 0]
    y = df.iloc[:, 1]

    coefficients = np.polyfit(x, y, 1)
    polynomial = np.poly1d(coefficients)
    y_fit = polynomial(x)

    plt.figure(figsize=(10, 6))
    plt.plot(x, y, "o", label="Data Points")
    plt.plot(x, y_fit, "-", label="Best Fit Line")
    plt.title("Line Chart with Best Fit Line")
    plt.xlabel(df.columns[0])
    plt.ylabel(df.columns[1])
    plt.legend()
    plt.grid(True)
    plt.show()


# Define the function to execute the tools
def execute_tool(tool_calls, messages):
    for tool_call in tool_calls:
        tool_name = tool_call.function.name
        tool_arguments = json.loads(tool_call.function.arguments)
        ic(tool_name, tool_arguments)

        if tool_name == "CleanDataTool":
            # Simulate data cleaning
            cleaned_df = clean_data(tool_arguments["data"])
            cleaned_data = {"cleaned_data": cleaned_df.to_dict()}
            messages.append(
                {"role": "tool", "name": tool_name, "content": json.dumps(cleaned_data)}
            )
            print("Cleaned data: ", cleaned_df)
        elif tool_name == "TransformDataTool":
            # Simulate data transformation
            transformed_data = {"transformed_data": "sample_transformed_data"}
            messages.append(
                {
                    "role": "tool",
                    "name": tool_name,
                    "content": json.dumps(transformed_data),
                }
            )
        elif tool_name == "AggregateDataTool":
            # Simulate data aggregation
            aggregated_data = {"aggregated_data": "sample_aggregated_data"}
            messages.append(
                {
                    "role": "tool",
                    "name": tool_name,
                    "content": json.dumps(aggregated_data),
                }
            )
        elif tool_name == "StatAnalysisTool":
            # Simulate statistical analysis
            stats_df = stat_analysis(tool_arguments["data"])
            stats = {"stats": stats_df.to_dict()}
            messages.append(
                {"role": "tool", "name": tool_name, "content": json.dumps(stats)}
            )
            print("Statistical Analysis: ", stats_df)
        elif tool_name == "CorrelationAnalysisTool":
            # Simulate correlation analysis
            correlations = {"correlations": "sample_correlations"}
            messages.append(
                {"role": "tool", "name": tool_name, "content": json.dumps(correlations)}
            )
        elif tool_name == "RegressionAnalysisTool":
            # Simulate regression analysis
            regression_results = {"regression_results": "sample_regression_results"}
            messages.append(
                {
                    "role": "tool",
                    "name": tool_name,
                    "content": json.dumps(regression_results),
                }
            )
        elif tool_name == "CreateBarChartTool":
            # Simulate bar chart creation
            bar_chart = {"bar_chart": "sample_bar_chart"}
            messages.append(
                {"role": "tool", "name": tool_name, "content": json.dumps(bar_chart)}
            )
        elif tool_name == "CreateLineChartTool":
            # Simulate line chart creation
            line_chart = {"line_chart": "sample_line_chart"}
            messages.append(
                {"role": "tool", "name": tool_name, "content": json.dumps(line_chart)}
            )
            plot_line_chart(tool_arguments["data"])
        elif tool_name == "CreatePieChartTool":
            # Simulate pie chart creation
            pie_chart = {"pie_chart": "sample_pie_chart"}
            messages.append(
                {"role": "tool", "name": tool_name, "content": json.dumps(pie_chart)}
            )
    return messages


# Define the functions to handle each agent's processing
def handle_data_processing_agent(query, conversation_messages):
    messages = [{"role": "system", "content": processing_system_prompt}]
    messages.append({"role": "user", "content": query})

    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0,
        tools=[
            pydantic_function_tool(CleanDataTool),
            pydantic_function_tool(TransformDataTool),
            pydantic_function_tool(AggregateDataTool),
        ],
    )

    conversation_messages.append(
        [tool_call.function for tool_call in response.choices[0].message.tool_calls]
    )
    execute_tool(response.choices[0].message.tool_calls, conversation_messages)


def handle_analysis_agent(query, conversation_messages):
    messages = [{"role": "system", "content": analysis_system_prompt}]
    messages.append({"role": "user", "content": query})

    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0,
        tools=[
            pydantic_function_tool(StatAnalysisTool),
            pydantic_function_tool(CorrelationAnalysisTool),
            pydantic_function_tool(RegressionAnalysisTool),
        ],
    )

    conversation_messages.append(
        [tool_call.function for tool_call in response.choices[0].message.tool_calls]
    )
    execute_tool(response.choices[0].message.tool_calls, conversation_messages)


def handle_visualization_agent(query, conversation_messages):
    messages = [{"role": "system", "content": visualization_system_prompt}]
    messages.append({"role": "user", "content": query})

    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0,
        tools=[
            pydantic_function_tool(CreateBarChartTool),
            pydantic_function_tool(CreateLineChartTool),
            pydantic_function_tool(CreatePieChartTool),
        ],
    )

    conversation_messages.append(
        [tool_call.function for tool_call in response.choices[0].message.tool_calls]
    )
    execute_tool(response.choices[0].message.tool_calls, conversation_messages)


# Function to handle user input and triaging
def handle_user_message(user_query, conversation_messages=[]):
    user_message = {"role": "user", "content": user_query}
    conversation_messages.append(user_message)

    messages = [{"role": "system", "content": triaging_system_prompt}]
    messages.extend(conversation_messages)

    completion = client.beta.chat.completions.parse(
        model=MODEL,
        messages=messages,
        temperature=0,
        tools=[pydantic_function_tool(Triage)],
    )

    tool_calls = completion.choices[0].message.tool_calls
    conversation_messages.append([tool_call.function for tool_call in tool_calls])

    # print number of tool_calls
    ic(len(tool_calls))

    for tool_call in tool_calls:
        function_name = tool_call.function.name
        function_arguments = tool_call.function.arguments
        ic(function_name, function_arguments)
        if function_name == "Triage":
            agents = json.loads(tool_call.function.arguments)["agents"]
            query = json.loads(tool_call.function.arguments)["query"]
            ic(agents, query)
            for agent in agents:
                if agent == "Data Processing Agent":
                    handle_data_processing_agent(query, conversation_messages)
                elif agent == "Analysis Agent":
                    handle_analysis_agent(query, conversation_messages)
                elif agent == "Visualization Agent":
                    handle_visualization_agent(query, conversation_messages)

    return conversation_messages


handle_user_message(user_query)
