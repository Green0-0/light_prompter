from model import Chat
from responder import Responder, Response


class Critique_LLM(Responder):
    """
    A responder that generates a response, critiques it, and then generates a new response based on the critique.
    """
    def __init__(self, response_responder: Responder, critique_responder: Responder, refine_responder: Responder, prompt_critique: str, prompt_refine: str):
        """
        Initializes a Critique_LLM object.

        Args:
            response_responder (Responder): The responder to generate the initial response.
            critique_responder (Responder): The responder to generate the critique.
            refine_responder (Responder): The responder to generate the refined response.
        """
        self.response_responder = response_responder
        self.critique_responder = critique_responder
        self.refine_responder = refine_responder
        self.prompt_critique = prompt_critique
        self.prompt_refine = prompt_refine
        self.response = None
        self.critique = None
        self.refined_response = None

    def start(self, prompt: str) -> list[list[Chat]]:
        """
        Calls start on the response responder, returns a list of lists of chat requests.

        Args:
            prompt (str): The prompt to respond to.

        Returns:
            list[list[Chat]]: A list of lists of Chat objects from the response responder.
        """
        self.prompt = prompt
        self.response = None
        self.critique = None
        self.refined_response = None
        chats = [self.response_responder.start(prompt)]
        return chats

    def step(self, inputs: list[list[str]]) -> list[list[Chat]] | None:
        """
        Forwards completions to the appropriate responders, saves finished responders. Returns None when all responders are finished.

        Args:
            inputs (list[list[str]]): A list of lists of model completions.

        Returns:
            list[list[Chat]] | None: A list of lists of Chat objects to be completed or None if finished.
        """
        if self.response is None:
            chats = [self.response_responder.step(inputs[0])]
            if chats[0] is None:
                self.response = self.response_responder.finish()
                critic_prompt = self.prompt_critique.replace("{{query}}", self.prompt).replace("{{response}}", self.response.output)
                chats = [self.critique_responder.start(critic_prompt)]
            return chats
        elif self.critique is None:
            chats = [self.critique_responder.step(inputs[0])]
            if chats[0] is None:
                self.critique = self.critique_responder.finish()
                refine_prompt = self.prompt_refine.replace("{{query}}", self.prompt).replace("{{response}}", self.response.output).replace("{{critique}}", self.critique.output)
                chats = [self.refine_responder.start(refine_prompt)]
            return chats
        else:
            chats = [self.refine_responder.step(inputs[0])]
            if chats[0] is None:
                self.refined_response = self.refine_responder.finish()
                return None
            return [chats]

    def finish(self) -> Response:
        """
        Returns the final refined response.

        Returns:
            Response: A Response object containing the final response.
        """
        verbose_output = f"Initial Response:\n{self.response.verbose_output}\n-------\n"
        verbose_output += f"Critique:\n{self.critique.verbose_output}\n-------\n"
        verbose_output += f"Refined Response:\n{self.refined_response.verbose_output}"
        return Response(verbose_output=verbose_output, output=self.refined_response.output, answer=self.refined_response.answer)
