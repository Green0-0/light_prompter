import requests
import time
from typing import List

class Chat:
    system_prompt: str # system prompt
    messages: list[str] # list of messages in this chat
    max_retrieves: int # Number of messages to retrieve on toOAI

    def __init__(self, message : str = "", system_prompt = "", cloned_chat : "Chat" = None, max_retrieves : int = 10):
        """
        Create a new chat with an optional starting message and an optional system prompt.
        Alternatively, copy from another Chat instance.

        Args:
            message (str): optional starting message for the chat
            system_prompt (str): optional system prompt for the chat
            cloned_chat (Chat): optional Chat object to copy from
            max_retrieves (int): optional maximum number of messages to retrieve
        """
        if cloned_chat is not None:
            # Copy constructor behavior
            self.messages = cloned_chat.messages.copy()
            self.system_prompt = cloned_chat.system_prompt
            self.max_retrieves = cloned_chat.max_retrieves
        else:
            # Standard constructor behavior
            if message == "":
                self.messages = []
            else:
                self.messages = [message]
            self.system_prompt = system_prompt
            self.max_retrieves = max_retrieves
    
    def append(self, message : str):
        """Append message to the end of the message list, making it the new final message."""
        self.messages.append(message)
    
    def appendContinue(self, message : str):
        """Append message to the end of the final message in the message list."""
        self.messages[-1] += message
        
    def toOAI(self):
        """
        Convert the message list to an OpenAI API compatible list of messages.

        Returns:
            list[dict]: List of messages formatted for OpenAI API.
        """
        oai_messages = []
        if self.system_prompt != "":
            oai_messages.append({"role": "system", "content": self.system_prompt})
        for i, msg in enumerate(self.messages):
            role = "user" if i % 2 == 0 else "assistant"
            oai_messages.append({"role": role, "content": msg})
        while len(oai_messages) > self.max_retrieves:
            oai_messages.pop(0)
            oai_messages.pop(0)
        return oai_messages
    
    def __str__(self):
        """
        Overload the __str__ method to convert the Chat object to a string representation.

        Returns:
            str: String representation of the Chat object.
        """
        messages = ["System Prompt: " + self.system_prompt]
        for i, message in enumerate(self.messages):
            role = "User" if i % 2 == 0 else "Assistant"
            messages.append(f"{role}: {message}")
        return "\n".join(messages)

class Sampler:
    temperature : float
    top_p : float
    frequency_penalty : float
    max_tokens : int

    def __init__(self, temperature : float = 0.8, top_p : float = 0.9, frequency_penalty : float = 0, max_tokens : int = 4096):
        """
        Initialize a Sampler instance with optional parameters for controlling
        the behavior of text generation.

        Args:
            temperature (float, optional): Controls randomness in predictions. 
                Higher values result in more random completions. Defaults to 0.8.
            top_p (float, optional): Probability threshold for nucleus sampling.
                Defaults to 0.9.
            frequency_penalty (float, optional): Penalty for repeated tokens.
                Defaults to 0.
            max_tokens (int, optional): Maximum number of new tokens to generate.
                Defaults to 1024.
        """
        self.temperature = temperature
        self.top_p = top_p
        self.frequency_penalty = frequency_penalty
        self.max_tokens = max_tokens
        
class Model:
    url: str # URL of the OpenAI API
    api_key: str # API key for the OpenAI API
    model: str # type of model to use for the completion
    sampler : Sampler
    
    def __init__(self, url : str = None, model : str = None, api_key : str = None, sampler : Sampler = Sampler()):
        """
        Initialize a Model instance with optional parameters for URL, model type, and API key.

        Args:
            url (str, optional): The URL of the OpenAI API. Defaults to None.
            model (str, optional): The type of model to use for completion. Defaults to None.
            api_key (str, optional): The API key for authenticating with the OpenAI API. Defaults to None.
            sampler (Sampler, optional): The Sampler object for controlling the behavior of text generation. Defaults to Sampler().
        """
        self.url = url
        if self.url[-1] == '/':
            self.url = self.url[:-1]
        self.model = model
        self.api_key = api_key
        self.sampler = sampler
        
    def get_reply(self, input : "Chat", timeout_duration : int = 180, max_retries : int = 3) -> str:        
        """
        Get the completion of a given chat.

        Args:
            input (Chat): The OpenAI API compatible list of messages.
            timeout_duration (int, optional): Timeout in seconds for the API request. Defaults to 180.
            max_retries (int, optional): Maximum number of retries for the API request. Defaults to 3.

        Returns:
            str: The completion of the chat.

        Raises:
            requests.Timeout: If the request times out
            requests.RequestException: For other request-related errors
        """
        
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        data = {
            'model': self.model,
            'messages': input.toOAI(),
            'stream': False,
            'temperature': self.sampler.temperature,
            'top_p': self.sampler.top_p,
            'frequency_penalty': self.sampler.frequency_penalty,
            'max_tokens': self.sampler.max_tokens,
        }
        
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                response = requests.post(self.url + '/chat/completions', headers=headers, json=data, timeout=timeout_duration)
                response.raise_for_status()  # Raise an exception for bad status codes
                response = response.json()
                try: 
                    response = response['choices'][0]['message']['content']
                    return response
                except KeyError:
                    print("Response does not have the expected format: ", response)
            except requests.Timeout:
                print(f"Request timed out, retrying ({retry_count}/{max_retries})...")
            except requests.RequestException as e:
                print(f"Error during request: {str(e)}")
                print(f"Retrying ({retry_count}/{max_retries})...")
            retry_count += 1
            time.sleep(2)
        
        raise requests.RequestException("Max retries exceeded")
        
    def get_reply_batch(self, inputs : list["Chat"], timeout_duration : int = 180, max_retries : int = 3) -> list[str]:
        replies = []
        for input in inputs:
            replies.append(self.get_reply(input, timeout_duration, max_retries))
        return replies
        
    def __str__(self):
        """
        Converts the Model object to a string representation.

        Returns:
            str: String representation of the Model object.
        """
        return f"Model(url: {self.url}, model name: {self.model}, api_key: {self.api_key})"
    
class VLLMModel:
    def __init__(self, model, sampler) -> None:
        self.model = model
        self.sampler = sampler

    def get_reply(self, input : "Chat", timeout_duration : int = 180, max_retries : int = 3) -> str:
        tokenizer = self.model.get_tokenizer()
        input_oai = input.toOAI()
        formatted_inputs = [tokenizer.apply_chat_template(input_oai, tokenize=False, add_generation_prompt=True)]

        responses = self.model.generate(formatted_inputs, self.sampler)
        return responses[0].outputs[0].text

    def get_reply_batch(self, inputs : list["Chat"], timeout_duration : int = 180, max_retries : int = 3) -> list[str]:
        tokenizer = self.model.get_tokenizer()
        input_oai = [input.toOAI() for input in inputs]
        formatted_inputs = [tokenizer.apply_chat_template(input_oai[i], tokenize=False, add_generation_prompt=True) for i in range(len(input_oai))]

        responses = self.model.generate(formatted_inputs, self.sampler)
        return [response.outputs[0].text for response in responses]