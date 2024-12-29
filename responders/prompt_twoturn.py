from model import Chat
from responder import Responder, Response

from typing import Callable

class Prompt_TwoTurn(Responder):
    """
    A responder that uses a prompt to generate a response and then extracts the answer.
    """
    def __init__(self, prompt_get_answer: str = "Now output the answer, and only the answer, without any other formatting.", answer_extractor: Callable[[str], str] = lambda x: x):
        """
        Initializes a Prompt_TwoTurn object.

        Args:
            prompt_get_answer (str): The prompt to use for extracting the answer.
            answer_extractor (Callable[[str], str]): Function to extract answer from model output.
        """
        self.prompt_get_answer = prompt_get_answer
        self.answer_extractor = answer_extractor
        self.answer = ""
        self.chat = None

    def start(self, prompt: str) -> list[Chat]:
        """
        Creates an initial array of one chat request using the prompt.

        Args:
            prompt (str): The prompt to use for generating the response.

        Returns:
            list[Chat]: A list containing the initial Chat object.
        """
        self.answer = ""
        self.chat = Chat(message=prompt)
        return [self.chat]

    def step(self, inputs: list[str]) -> list[Chat] | None:
        """
        Saves the response to intermediate_value, returns a new chat request asking the model to output only the answer.
        On the second call, returns None and saves the response as the second response.

        Args:
            inputs (list[str]): A list containing the model completion.

        Returns:
            list[Chat] | None: A list containing the new Chat object or None if finished.
        """
        if len(self.chat.messages) == 1:
            self.intermediate_value = inputs[0]
            self.chat.append(self.intermediate_value)
            self.chat.append(self.prompt_get_answer)
            return [self.chat]
        else:
            self.answer = inputs[0]
            return None

    def finish(self) -> Response:
        """
        Returns the Response.

        Returns:
            Response: A Response object containing the final response.
        """
        answer = self.answer_extractor(self.answer)
        return Response(verbose_output=self.intermediate_value + "\n-------\n" + self.answer, output=self.intermediate_value, answer=answer)
