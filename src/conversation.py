from rich import print

from chat import Chat


class Persona:
    def __init__(self, role: str, purpose: str):
        self.role = role
        self.purpose = purpose
        self.chat = Chat()
        self.chat.ask(f"You are the {role} persona. Your purpose is to {purpose}")

    def summarize(self, argument: str, paragraphs: int = 3):
        _query = f"""{argument}
        Please summarize the above in {paragraphs} cohesive paragraphs
        
        --ouput format start--
        List of 
            Main Point: str
            Elaboration: str
        --output format end--
        """
        return self.chat.ask(_query)

    def research(self, argument: str):
        query = f"Please use Browser to find relevant information for {argument}"
        return self.chat.browser_results(query)

    def rebut(self, argument: str, history: str):
        argument_summary = self.chat.ask(f"Summarize this argument: {argument}")
        negative_statement = self.chat.ask(
            f"Please provide a negative statement for the argument: {argument_summary}"
        )
        _query = f"""{self.get_history(history)}
        {negative_statement}
        """
        rebuttal = self.research(_query)
        return self.summarize(rebuttal)

    def get_history(self, history: str):
        return f"""Here is the history of the debate:
        --history-start--
        {history}
        --history-end--
        """

    def argue(self, history: str):
        _query = f"""{self.get_history(history)}
        Please present new arguments as the {self.role}. 
        
        Do not repeat previous arguments
        
        --output format start---
        List of 
            argument_main_point: str
            argument_elaboration: str 
        --output format end--
        """
        argument = self.chat.ask(_query)
        _argument = self.research(argument)
        return self.summarize(_argument)


class Conversation:
    def __init__(self, topic: str):
        self.topic = topic
        self.history = []
        self.proposer = Persona("proposer", f"argue for the topic: {topic}")
        self.opposer = Persona("opposer", f"argue against the topic: {topic}")

    def get_history(self):
        return "\n".join(self.history)

    def append_to_history(self, message: str):
        print(message)
        self.history.append(message)

    def converse(self, rounds: int = 5):
        argument = ""
        while rounds > 0:
            if len(self.history) > 0:
                argument = self.proposer.rebut(argument, self.get_history())
                self.append_to_history(f"Proposer: {argument}")
            argument = self.proposer.argue(self.get_history())
            self.append_to_history(f"Proposer: {argument}")
            argument = self.opposer.rebut(argument, self.get_history())
            self.append_to_history(f"Opposer: {argument}")
            argument = self.opposer.argue(self.get_history())
            self.append_to_history(f"Opposer: {argument}")

            rounds -= 1


def main():
    topic = "WordPress is better than Wix"
    topic = "A millionaire is happier than a billionaire"
    conversation = Conversation(topic)
    conversation.converse()


if __name__ == "__main__":
    main()
