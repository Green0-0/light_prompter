from model import Chat
from responder import Responder, Response


from typing import Callable


class PickCommon_Custom(Responder):
    """
    A responder that picks the most common answer from a list of responders, using a custom similarity function.
    """
    def __init__(self, responders: set[Responder], similarity_function: Callable[[str, str], float] = lambda a, b: 1.0 if a == b else 0.0):
        """
        Initializes a PickCommon_Custom object.

        Args:
            responders (set[Responder]): A set of Responder objects.
            similarity_function (Callable[[str, str], float]): A function that takes two strings and returns a float representing their similarity.
        """
        self.responders = set(responders)
        self.active_responders = []
        self.responses = []
        self.similarity_function = similarity_function

    def start(self, prompt: str) -> list[list[Chat]]:
        """
        Calls start on all responders, returns a list of lists of chat requests.

        Args:
            prompt (str): The prompt to respond to.

        Returns:
            list[list[Chat]]: A list of lists of Chat objects from all responders.
        """
        self.active_responders = list(self.responders)
        self.responses = []
        chats = []
        for responder in self.active_responders:
            chats.append(responder.start(prompt))
        return chats

    def step(self, inputs: list[list[str]]) -> list[list[Chat]] | None:
        """
        Forwards completions to responders, saves finished responders. Returns None when all responders are finished.

        Args:
            inputs (list[list[str]]): A list of lists of model completions.

        Returns:
            list[list[Chat]] | None: A list of lists of Chat objects to be completed or None if finished.
        """
        to_delete = []
        chats = []
        for i, responder in enumerate(self.active_responders):
            result = responder.step(inputs[i])
            if result is None:
                self.responses.append(responder.finish())
                to_delete.append(responder)
            else:
                chats.append(result)

        for responder in to_delete:
            self.active_responders.remove(responder)

        if not self.active_responders:
            return None
        return chats

    def finish(self) -> Response:
        """
        Chooses the answer with the highest average similarity to all other answers.

        Returns:
            Response: A Response object containing the final response.
        """
        answers = [response.answer for response in self.responses]

        similarity_scores = {}
        for i, answer1 in enumerate(answers):
            similarity_scores[i] = 0
            for j, answer2 in enumerate(answers):
                if i != j:
                    similarity_scores[i] += self.similarity_function(answer1, answer2)

        most_similar_index = max(similarity_scores, key=similarity_scores.get)
        most_similar_response = self.responses[most_similar_index]

        verbose_output = ""
        for response in self.responses:
            verbose_output += response.verbose_output + "\n-------\n"
        verbose_output += "\n-------\nMost similar answer: " + str(most_similar_response.answer)

        return Response(verbose_output=verbose_output, output=most_similar_response.output, answer=most_similar_response.answer)
