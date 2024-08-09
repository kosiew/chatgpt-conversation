import json
from enum import Enum
from io import StringIO
from re import A
from typing import List, Optional

from duckduckgo_search import DDGS
from icecream import ic

# openai's BaseModel require Config.extra = "forbid" to ensure no additional properties are allowed
# we're also using Config to add descriptions to the pydantic models
from openai import BaseModel, OpenAI, pydantic_function_tool
from pydantic import Field, HttpUrl, ValidationError

from structured_output.main import completions, test_response_format
from structured_output.multi_agent import BaseTool, SpeakToUserTool


class SearchResult(BaseModel):
    url: Optional[str]
    title: str
    snippet: Optional[str]


client = OpenAI()
MODEL = "gpt-4o-2024-08-06"
duck = DDGS()

# system prompts for agents
triaging_system_prompt = """You are a Triaging Agent. Your role is to assess the user's query and route it to the relevant agents. The agents available are:
- Browsing Agent: Searches the web for information.
- Analysis Agent: Analyzes and summarizes information
Use the TriageTool to forward the user's query to the relevant agents. Also, use the SpeakToUserTool to get more information from the user if needed."""

browse_system_prompt = """You are a Browsing Agent. Your role is to search the web for information based on the user's query. Use the SearchTool to find relevant information and provide the user with the necessary details."""

analysis_system_prompt = """You are an Analysis Agent. Your role is to analyze and summarize information obtained from the web based on the user's query. Use the AnalyzeTool to process the information and provide the user with a concise summary."""


class Agent(str, Enum):
    browsing = "browsing"
    analysis = "analysis"


class TriageTool(BaseTool):
    agents: list[Agent] = Field(
        ..., description="The list of agents to route the query to."
    )
    query: str = Field(..., description="The user's query to be triaged.")

    class Config:
        description = "Routes the user's query to the relevant agents for processing."


class SearchTool(BaseTool):
    query: str = Field(..., description="The user's query to search for on the web.")

    class Config:
        description = "Searches the web for information based on the user's query."


class AnalyzeTool(BaseTool):
    content: str = Field(..., description="The content to be analyzed and summarized.")

    class Config:
        description = "Analyzes and summarizes information based on the user's query."


def search_web(query: str) -> List[SearchResult]:
    results = duck.text(query, backend="api")
    collected_results = []
    for d in results:
        result = {}
        if not isinstance(d, dict):
            continue
        try:
            ic(d)
            url = d.get("href", None)
            title = d.get("title", "no title")
            snippet = d.get("body", None)
            if not all([url, title, snippet]):
                raise ValueError("Missing required fields in duckduckgo search result")
            result = SearchResult(url=url, title=title, snippet=snippet)
            collected_results.append(result)
        except Exception as e:
            print(f"Error occurred when processing {result=}: {e}")
    return collected_results


class ToolHandler:
    def __init__(self):
        self.tool_functions = {
            "SearchTool": self.search_tool,
            "AnalzzeTool": self.analyze_tool,
        }

    def search_tool(self, arguments, messages):
        query = arguments.get("query")
        results = search_web(query)
        for result in results:
            messages.append(
                {
                    "role": "tool",
                    "name": "SearchTool",
                    "content": f"Title: {result.title}\nURL: {result.url}\nSnippet: {result.snippet}",
                }
            )

    def analyze_tool(self, arguments, messages):
        content = arguments.get("content")
        response = client.completions(
            model=MODEL,
            prompt=f"Analyze the following content and provide a summary:\n{content}",
        )
        messages.append(
            {
                "role": "tool",
                "name": "AnalyzeTool",
                "content": response.choices[0].text,
            }
        )


class Summary(BaseTool):
    summary: str = Field(..., description="The summary of the analyzed content.")

    class Config:
        description = "The summary of the analyzed content."


def get_summary(_messages):
    messages = [
        {"role": "system", "content": "Summarize the content."},
        {
            "role": "user",
            "content": "Alice and Bob are going to a science fair on Friday.",
        },
    ]
    test_response_format(messages, completions, Summary)
