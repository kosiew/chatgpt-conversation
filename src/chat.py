import json
import os
from typing import Dict, List

from openai import OpenAI

# type alias for Dict[str, str]
Response = Dict[str, str]


class Chat:
    def __init__(
        self,
        openai_api_key: str = None,
        model: str = "gpt-4o",
        json_output: bool = True,
    ):
        """
        Initializes the Chat class with API key and model.
        """
        api_key = None
        if not openai_api_key and not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError(
                "You must supply openai_api_key or set environment variable OPENAI_API_KEY"
            )
        elif openai_api_key:
            api_key = openai_api_key
        else:
            api_key = os.environ["OPENAI_API_KEY"]

        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.chat_history: List[Response] = []
        if json_output:
            self.response_format = {"type": "json_object"}
            self.chat_history.append(
                {
                    "role": "system",
                    "content": "Set the response format to JSON object",
                }
            )
        else:
            self.response_format = {"type": "text"}

    def append_response_to_history(self, response: Response):
        message = response.choices[0].message

        self.chat_history.append({"role": message.role, "content": message.content})

    def get_reply(self, response: Response):
        content = response.choices[0].message.content
        return content

    def print_reply(self, response: Response):
        print(self.get_reply(response))

    def ask(self, query):
        self.chat_history.append({"role": "user", "content": query})

        response = self.client.chat.completions.create(
            model=self.model,
            response_format=self.response_format,
            messages=self.chat_history,
        )

        self.append_response_to_history(response)

        # self.print_reply(response)
        print(response.usage.total_tokens)
        return self.get_reply(response)

    def browser_results(self, query: str, k: int = 3):
        _query = f"Use Browser to retrieve {k} results related to {query}. The output should be a results array of dict of (title, url, description, snippets(array of string))"
        json_content = self.ask(_query)
        return json.loads(json_content)
