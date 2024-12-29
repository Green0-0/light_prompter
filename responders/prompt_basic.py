from light_prompter.model import Chat
from light_prompter.responder import Responder, Response

from typing import Callable

class Prompt_Basic(Responder):
    """
    A responder that uses a prompt to generate a response.
    """
    def __init__(self, answer_extractor: Callable[[str], str] = lambda x: x):
        """
        Initializes a Prompt_Basic object.

        Args:
            answer_extractor (Callable[[str], str]): Function to extract answer from model output.
        """
        self.answer = ""
        self.answer_extractor = answer_extractor

    def start(self, prompt: str) -> list[Chat]:
        """
        Creates an initial array of one chat request using the prompt.

        Args:
            prompt (str): The prompt to use for generating the response.

        Returns:
            list[Chat]: A list containing the initial Chat object.
        """
        self.answer = ""
        return [Chat(message=prompt)]

    def step(self, inputs: list[str]) -> None:
        """
        Takes an array of one completed request, always returns None.

        Args:
            inputs (list[str]): A list containing the model completion.

        Returns:
            None: Always returns None.
        """
        self.answer = inputs[0]
        return None

    def finish(self) -> Response:
        """
        Returns the Response.

        Returns:
            Response: A Response object containing the final response.
        """
        answer = self.answer_extractor(self.answer)
        return Response(verbose_output=self.answer, output=self.answer, answer=answer)