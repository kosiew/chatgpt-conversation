import json
import os

from openai import OpenAI


class Chat:
    def __init__(self, openai_api_key=None, model="gpt-4o"):
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
        self.chat_history = [
            {
                "role": "system",
                "content": "You are a helpful assistant designed to output JSON.",
            }
        ]

    def append_response_to_history(self, response):
        message = response.choices[0].message

        self.chat_history.append({"role": message.role, "content": message.content})

    def get_reply(self, response):
        content = response.choices[0].message.content
        return content

    def print_reply(self, response):
        print(self.get_reply(response))

    def ask(self, query, model="gpt-4o"):
        self.chat_history.append({"role": "user", "content": query})

        response = self.client.chat.completions.create(
            model=model,
            response_format={"type": "json_object"},
            messages=self.chat_history,
        )

        self.append_response_to_history(response)

        # self.print_reply(response)
        print(response.usage.total_tokens)
        return self.get_reply(response)

    def browser_results(self, query, k=3):
        _query = f"Use Browser to retrieve {k} results related to {query}. The output should be a results array of dict of (title, url, description, snippets(array of string))"
        json_content = self.ask(_query)
        return json.loads(json_content)
        return json.loads(json_content)
