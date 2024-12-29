from model import Chat
from responder import Responder, Response


from typing import Callable


class Aggregate_LLM(Responder):
    """
    A responder that aggregates responses using another responder.
    """
    def __init__(self, responders: set[Responder], aggregation_responder: Responder, aggregation_prompt: str, final_answer_extractor: Callable[[str], str] = lambda x: x):
        """
        Initializes an Aggregate_LLM object.

        Args:
            responders (list[Responder]): A list of Responder objects.
            aggregation_responder (Responder): Responder to use for combining responses.
            answer_extractor (Callable[[str], str]): Function to extract answer from model output.
        """
        self.responders = set(responders)
        self.aggregation_responder = aggregation_responder
        self.active_responders = []
        self.responses = []
        self.final_response = None
        self.prompt_combine = aggregation_prompt
        self.prompt = ""
        self.final_answer_extractor = final_answer_extractor

    def start(self, prompt: str) -> list[list[Chat]]:
        """
        Calls start on all responders, returns a list of lists of chat requests.

        Args:
            prompt (str): The prompt to respond to.

        Returns:
            list[list[Chat]]: A list of lists of Chat objects from all responders.
        """
        self.prompt = prompt
        self.active_responders = list(self.responders)
        self.responses = []
        self.final_response = None
        chats = []
        for responder in self.active_responders:
            chats.append(responder.start(prompt))
        return chats

    def step(self, inputs: list[list[str]]) -> list[list[Chat]] | None:
        """
        Forwards completions to responders, saves finished responders. After all responders have given a response,
        starts the aggregation responder with the combined prompt.

        Args:
            inputs (list[list[str]]): A list of lists of model completions.

        Returns:
            list[list[Chat]] | None: A list of lists of Chat objects to be completed or None if finished.
        """
        if self.active_responders:
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
                combined_responses = "\n\n".join([f"RESPONSE {i + 1}:\n{response.output}" for i, response in enumerate(self.responses)])
                combine_prompt = self.prompt_combine.replace("{{query}}", self.prompt).replace("{{responses}}", combined_responses)
                self.aggregation_chats = self.aggregation_responder.start(combine_prompt)
                return [self.aggregation_chats]
            return chats
        else:
            chats = [self.aggregation_responder.step(inputs[0])]
            if chats[0] is None:
                self.final_response = self.aggregation_responder.finish()
                return None
            return chats

    def finish(self) -> Response:
        """
        Returns the combined response.

        Returns:
            Response: A Response object containing the final response.
        """
        verbose_output = ""
        for response in self.responses:
            verbose_output += response.verbose_output + "\n-------\n"
        verbose_output += "\n-------\nAGGREGATION:\n" + str(self.final_response.verbose_output)
        final_final_answer = self.final_answer_extractor(self.final_response.answer) if self.final_answer_extractor else self.final_response.answer
        return Response(verbose_output=verbose_output, output=self.final_response.answer, answer=final_final_answer)
