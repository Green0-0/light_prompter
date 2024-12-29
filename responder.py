from typing import Any, List
from model import Model, Chat
from collections import Counter

class Response:
    """
    A class to represent a response from a responder.

    Attributes:
        verbose_output (str): Detailed output of the response, including all steps.
        output (str): Actual output which led to the answer.
        answer (str): The answer to the question.
    """
    def __init__(self, verbose_output: str, output: str, answer: str):
        """
        Initializes a Response object.

        Args:
            verbose_output (str): Detailed output of the response, including all steps.
            output (str): Actual output which led to the answer.
            answer (str): The answer to the question.
        """
        self.verbose_output = verbose_output
        self.output = output
        self.answer = answer

class Responder:
    """
    A base class for responders.
    """
    def start(self, prompt: str) -> list[Chat]:
        """
        Sets up the model and returns an array of chats to be completed.

        Args:
            prompt (str): The prompt to respond to.

        Returns:
            list[Chat]: A list of Chat objects to be completed.
        """
        raise NotImplementedError

    def step(self, inputs: list[str]) -> list[Chat] | None:
        """
        Processes the input array and returns either an array of chats or None if finished.

        Args:
            inputs (list[str]): A list of model completions.

        Returns:
            list[Chat] | None: A list of Chat objects to be completed or None if finished.
        """
        raise NotImplementedError

    def finish(self) -> Response:
        """
        Returns the final response.

        Returns:
            Response: A Response object containing the final response.
        """
        raise NotImplementedError
    
def execute(model: Model, responder: Responder, prompt: str) -> Response:
    """
    Executes a responder and returns the final response.

    Args:
        responder (Responder): The responder to execute.
        prompt (str): The prompt to respond to.
        model (Model): The model to use for generating the response.

    Returns:
        Response: The final response from the responder.
    """
    def flatten(nested_list):
        flat_list = []
        for item in nested_list:
            if isinstance(item, list):
                flat_list.extend(flatten(item))
            else:
                flat_list.append(item)
        return flat_list

    def re_nest(flat_list, structure):
        nested_list = []
        i = 0
        for item in structure:
            if isinstance(item, list):
                sub_len = len(flatten([item]))
                nested_list.append(re_nest(flat_list[i:i+sub_len], item))
                i += sub_len
            else:
                nested_list.append(flat_list[i])
                i += 1
        return nested_list
    
    chats = responder.start(prompt)
    while chats:
        flat_chats = flatten(chats)
        replies = model.get_reply_batch(flat_chats)
        chats = responder.step(re_nest(replies, chats))
    return responder.finish()