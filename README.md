# Light Prompter - Scale test-time-compute with batching! 
Inspired by [optillm](https://github.com/codelion/optillm), Light Prompter is a Python framework for efficiently batching responses with prompting strategies that include multiple steps. This means that if you use an inference engine such as vLLM, you can get speedups of several magnitudes.

It works through a set of ``Responder``s, each of which acts as a state machine containing nested responders. Responders request a chat completion, which gets aggregated with all other chat completions so that they can be solved in one batch. Aftewards, these chat completions are sent back down to the responders.

Here's a diagram illustrating this process for something like self-consistency:

![diagram](https://github.com/user-attachments/assets/909eb3c1-bbf5-4e90-b72c-9565cae52147)

## Quick Start:
Use one of the following Kaggle notebooks:

1. https://www.kaggle.com/code/green000/light-prompter-fast-moa

This is an implementation of Mixture-Of-Agents from TogetherAI. Prompts were taken from https://github.com/togethercomputer/MoA.


2. https://www.kaggle.com/code/green000/light-prompter-fast-self-consistency

This is an implementation of self-consistency from https://arxiv.org/abs/2203.11171.

* Feel free to edit the similarity function as you wish, but be sure to include a case for when processing the answer fails, as it is not guranteed to be in the correct format.

## Slow Start:
Read the examples in the examples folder. 

Here are all the responders you can use:

**Prompt_Basic:** Single-step response generation

**Prompt_TwoTurn:** Two-step response and answer extraction

**Aggregate_LLM:** Combine multiple responses using another responder

**Critique_LLM:** Generates an initial response with a responder, critiques it with a second responder, and rewrites it using a third responder

**PickCommon_Custom:** Self-consistency; picks answer with most similarity to other answers based on the input function

All responders return a response which contain a verbose_output, an output, and an answer. The answer is the shortest specific answer, the output is the standard LLM output and the verbose_output contains all the steps taken.

Within ``sane_defaults.py`` you will find starter prompts (a basic COT prompt, the aggregator prompt used for MOA, a prompt that makes the aggregator choose the best response, a critique prompt, a rewriting prompt) and some answer extractors (which take in a string and try to "extract" the LLM's answer). Note that you can pass in answer extractor to most responders to try to parse the answer out. This is important! Some strategies like self-consistency rely on having a properly parsed answer.

After you have built a tree of responders, use the execute function from ``responders.py`` on the root. Make sure you have a model setup, the default in ``model.py`` is OpenAI compatible and thus will work with any local inference engine but is slow. The vLLM variant is much preferred. You can also implement your own.

Basic example:

```
from light_prompter import Model
from light_prompter import responder

from light_prompter.responders import Prompt_TwoTurn

# Initialize model
model = Model(url="https://api.example.com", api_key="your-api-key", model="model-name-here")

# Create responder
responder = Prompt_TwoTurn()

# Execute prompt
response = execute(model, responder, "What is the capital of France?")

print(response.output)
```

Contributions are welcome!

TODO:

1. Implement plansearch and code execution, possibly some kind of tree search
* it probably will not be possible to break plansearch down into a set of smaller responders like is done for MOA, as it is quite involved
2. Create a WebUI for testing with a list of preset responder configurations
3. Create an OpenAI compatible API

Planned:
1. Support more inference engines with batching out of the box such as exllama, sglang, tensorrtllm
2. A model router for routing requests to several API's at once when batching.
