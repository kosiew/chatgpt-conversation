import json
import os
from enum import Enum
from io import StringIO

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gtts import gTTS
from icecream import ic
from IPython.display import Image

# openai's BaseModel require Config.extra = "forbid" to ensure no additional properties are allowed
# we're also using Config to add descriptions to the pydantic models
from openai import BaseModel, OpenAI, pydantic_function_tool
from pydantic import Field

client = OpenAI()
MODEL = "gpt-4o-2024-08-06"

triaging_system_prompt = """You are a Triaging Agent. Your role is to assess the user's query and route it to the relevant agents. The agents available are:
- Data Processing Agent: Cleans, transforms, and aggregates data.
- Analysis Agent: Performs statistical, correlation, and regression analysis.
- Visualization Agent: Creates bar charts, line charts, and pie charts.
- Speak To User Agent: Convey message to user
Use the TriageTool to forward the user's query to the relevant agents."""

processing_system_prompt = """You are a Data Processing Agent. Your role is to clean, transform, and aggregate data using the following tools:
- CleanDataTool
- TransformDataTool
- AggregateDataTool"""

analysis_system_prompt = """You are an Analysis Agent. Your role is to perform statistical, correlation, and regression analysis on the data using the following tools:
- StatAnalysisTool
- CorrelationAnalysisTool
- RegressionAnalysisTool"""

visualization_system_prompt = """You are a Visualization Agent. Your role is to create bar charts, line charts, and pie charts of the data using the following tools:
- CreateBarChartTool
- CreateLineChartTool
- CreatePieChartTool
"""

speak_system_prompt = """You are a Speak To User Agent. Your role is to convey messages to the user, in a funny way, always include a Dad joke after the message, using SpeakToUserTool."""


class Agent(str, Enum):
    Data_Processing = "Data Processing Agent"
    Analysis = "Analysis Agent"
    Visualization = "Visualization Agent"
    SpeakToUser = "Speak To User Agent"


class BaseTool(BaseModel):
    class Config:
        extra = "forbid"


class TriageTool(BaseTool):
    agents: list[Agent] = Field(
        ..., description="The list of agents to route the query to."
    )
    query: str = Field(..., description="The user's query to be triaged.")

    class Config:
        description = "Routes the user's query to the relevant agents for processing."


class SpeakToUserTool(BaseTool):
    message: str = Field(..., description="The message to speak to the user.")

    class Config:
        description = "Sends a message to the user to request more information."


# Preprocessing tools
class CleanDataTool(BaseTool):

    data: str = Field(..., description="The input data to be cleaned.")

    class Config:
        description = "Cleans the provided data by removing duplicates and handling missing values."


class TransformDataTool(BaseTool):
    data: str = Field(..., description="The input data to be transformed.")
    rules: str = Field(
        ..., description="The transformation rules to be applied to the data."
    )

    class Config:
        description = "Transforms the provided data based on the specified rules."


class AggregateDataTool(BaseTool):
    data: str = Field(..., description="The input data to be aggregated.")
    group_by: list[str] = Field(..., description="The columns to group by.")
    operations: str = Field(..., description="The aggregation operations to perform.")

    class Config:
        description = "Aggregates the provided data based on the specified columns and operations."


# Analysis tools
class StatAnalysisTool(BaseTool):

    data: str = Field(..., description="The input data for statistical analysis.")

    class Config:
        description = "Performs statistical analysis on the provided dataset."


class CorrelationAnalysisTool(BaseTool):
    data: str = Field(..., description="The input data for correlation analysis.")
    variables: list[str] = Field(
        ..., description="The variables to calculate correlations for."
    )

    class Config:
        description = (
            "Calculates correlation coefficients between variables in the dataset."
        )


class RegressionAnalysisTool(BaseTool):
    data: str = Field(..., description="The input data for regression analysis.")
    dependent_var: str = Field(
        ..., description="The dependent variable for regression."
    )
    independent_vars: list[str] = Field(
        ..., description="The independent variables for regression."
    )

    class Config:
        description = "Performs regression analysis on the provided dataset."


# Visualization tools
class CreateBarChartTool(BaseTool):
    data: str = Field(..., description="The input data for the bar chart.")
    x: str = Field(..., description="The x-axis data.")
    y: str = Field(..., description="The y-axis data.")

    class Config:
        description = "Creates a bar chart from the provided data."


class CreateLineChartTool(BaseTool):
    data: str = Field(..., description="The input data for the line chart.")
    x: str = Field(..., description="The x-axis data.")
    y: str = Field(..., description="The y-axis data.")

    class Config:
        description = "Creates a line chart from the provided data."


class CreatePieChartTool(BaseTool):
    data: str = Field(..., description="The input data for the pie chart.")
    labels: str = Field(..., description="The labels for the pie chart.")
    values: str = Field(..., description="The values for the pie chart.")

    class Config:
        description = "Creates a pie chart from the provided data."


# Example query

user_query = """
Below is some data. I want you to perform these steps:
1. remove the duplicates 
2. analyze the statistics of the deduplicated data
3. plot a line chart on the deduplicated data.
4. message to user - "Good Day and Cowabunga!"

house_size (m3), house_price ($)
80, 90
80, 90
80, 90
90, 100
90, 100
100, 120
100, 120
100, 120
85, 95
70, 80
70, 80
70, 80
"""


def clean_data(data):
    data_io = StringIO(data)
    df = pd.read_csv(data_io, sep=",")
    df_deduplicated = df.drop_duplicates()
    return df_deduplicated


def speak_to_user(message):
    language = "en"
    speech = gTTS(text=message, lang=language, slow=False)
    speech.save("output.mp3")
    os.system("afplay output.mp3")


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


# Tool functions used in execute_tools


class ToolHandler:
    def __init__(self):
        self.tool_functions = {
            "TriageTool": self.triage_tool,
            "CleanDataTool": self.clean_data_tool,
            "SpeakToUserTool": self.speak_to_user_tool,
            "TransformDataTool": self.transform_data_tool,
            "AggregateDataTool": self.aggregate_data_tool,
            "StatAnalysisTool": self.stat_analysis_tool,
            "CorrelationAnalysisTool": self.correlation_analysis_tool,
            "RegressionAnalysisTool": self.regression_analysis_tool,
            "CreateBarChartTool": self.create_bar_chart_tool,
            "CreateLineChartTool": self.create_line_chart_tool,
            "CreatePieChartTool": self.create_pie_chart_tool,
        }

    def triage_tool(self, arguments, messages):
        agents = arguments["agents"]
        query = arguments["query"]
        ic(agents, query)
        for agent in agents:
            if agent == "Data Processing Agent":
                handle_data_processing_agent(query, messages)
            elif agent == "Analysis Agent":
                handle_analysis_agent(query, messages)
            elif agent == "Visualization Agent":
                handle_visualization_agent(query, messages)
            elif agent == "Speak To User Agent":
                handle_speak_to_user_agent(query, messages)

    def speak_to_user_tool(self, arguments, messages):
        message = arguments["message"]
        speak_to_user(message)
        messages.append({"role": "tool", "name": "SpeakToUserTool", "content": message})

    def clean_data_tool(self, arguments, messages):
        cleaned_df = clean_data(arguments["data"])
        cleaned_data = {"cleaned_data": cleaned_df.to_dict()}
        messages.append(
            {
                "role": "tool",
                "name": "CleanDataTool",
                "content": json.dumps(cleaned_data),
            }
        )
        ic("Cleaned data: ", cleaned_df)

    def transform_data_tool(self, arguments, messages):
        transformed_data = {"transformed_data": "sample_transformed_data"}
        messages.append(
            {
                "role": "tool",
                "name": "TransformDataTool",
                "content": json.dumps(transformed_data),
            }
        )

    def aggregate_data_tool(self, arguments, messages):
        aggregated_data = {"aggregated_data": "sample_aggregated_data"}
        messages.append(
            {
                "role": "tool",
                "name": "AggregateDataTool",
                "content": json.dumps(aggregated_data),
            }
        )

    def stat_analysis_tool(self, arguments, messages):
        stats_df = stat_analysis(arguments["data"])
        stats = {"stats": stats_df.to_dict()}
        messages.append(
            {"role": "tool", "name": "StatAnalysisTool", "content": json.dumps(stats)}
        )
        ic("Statistical Analysis: ", stats_df)

    def correlation_analysis_tool(self, arguments, messages):
        correlations = {"correlations": "sample_correlations"}
        messages.append(
            {
                "role": "tool",
                "name": "CorrelationAnalysisTool",
                "content": json.dumps(correlations),
            }
        )

    def regression_analysis_tool(self, arguments, messages):
        regression_results = {"regression_results": "sample_regression_results"}
        messages.append(
            {
                "role": "tool",
                "name": "RegressionAnalysisTool",
                "content": json.dumps(regression_results),
            }
        )

    def create_bar_chart_tool(self, arguments, messages):
        bar_chart = {"bar_chart": "sample_bar_chart"}
        messages.append(
            {
                "role": "tool",
                "name": "CreateBarChartTool",
                "content": json.dumps(bar_chart),
            }
        )

    def create_line_chart_tool(self, arguments, messages):
        line_chart = {"line_chart": "sample_line_chart"}
        messages.append(
            {
                "role": "tool",
                "name": "CreateLineChartTool",
                "content": json.dumps(line_chart),
            }
        )
        plot_line_chart(arguments["data"])

    def create_pie_chart_tool(self, arguments, messages):
        pie_chart = {"pie_chart": "sample_pie_chart"}
        messages.append(
            {
                "role": "tool",
                "name": "CreatePieChartTool",
                "content": json.dumps(pie_chart),
            }
        )

    # this is called in execute_tools
    def __call__(self, function_name, function_arguments, messages):
        if function_name in self.tool_functions:
            self.tool_functions[function_name](function_arguments, messages)
        else:
            ic(f"Unknown function: {function_name}")


def execute_tool_calls(tool_calls, tool_handler_cls, messages=[]):
    tool_calls_len = ic(len(tool_calls))
    tool_handler = tool_handler_cls()
    messages.append([tool_call.function for tool_call in tool_calls])
    for i, tool_call in enumerate(tool_calls):
        function_name = tool_call.function.name
        function_arguments = json.loads(tool_call.function.arguments)
        index = f"{i+1} of {tool_calls_len}"
        ic(index, function_name, function_arguments)
        tool_handler(function_name, function_arguments, messages)

    return messages


# Define a helper function to handle agent processing
def handle_agent(query, conversation_messages, system_prompt, tools):
    messages = [{"role": "system", "content": system_prompt}]
    messages.append({"role": "user", "content": query})

    # response = client.chat.completions.create also works
    response = client.beta.chat.completions.parse(
        model=MODEL,
        messages=messages,
        temperature=0,
        tools=[pydantic_function_tool(tool) for tool in tools],
    )

    tool_calls = response.choices[0].message.tool_calls
    if tool_calls:
        execute_tool_calls(tool_calls, ToolHandler, conversation_messages)


# Define the functions to handle each agent's processing
def handle_data_processing_agent(query, conversation_messages):
    handle_agent(
        query,
        conversation_messages,
        processing_system_prompt,
        [CleanDataTool, TransformDataTool, AggregateDataTool],
    )


def handle_analysis_agent(query, conversation_messages):
    handle_agent(
        query,
        conversation_messages,
        analysis_system_prompt,
        [StatAnalysisTool, CorrelationAnalysisTool, RegressionAnalysisTool],
    )


def handle_visualization_agent(query, conversation_messages):
    handle_agent(
        query,
        conversation_messages,
        visualization_system_prompt,
        [CreateBarChartTool, CreateLineChartTool, CreatePieChartTool],
    )


def handle_speak_to_user_agent(query, conversation_messages):
    handle_agent(
        query,
        conversation_messages,
        speak_system_prompt,
        [SpeakToUserTool],
    )


# Function to handle user input and triaging
def handle_user_query(user_query, conversation_messages=[]):
    handle_agent(
        user_query,
        conversation_messages,
        triaging_system_prompt,
        [TriageTool],
    )
    return conversation_messages


conversation_messages = handle_user_query(user_query)
ic(conversation_messages)
