from model import Model
import responder

from Responders.prompt_basic import Prompt_Basic
from Responders.prompt_twoturn import Prompt_TwoTurn
from Responders.pickcommon_custom import PickCommon_Custom
from Responders.aggregate_llm import Aggregate_LLM
from Responders.critique_llm import Critique_LLM

from sane_defaults import *

# Initialize model
model = Model(url="openai_url_here", api_key="api_key_here", model="model_name_here")

# Layer 1:
aggregator_prompt = get_aggregate_prompt()

aggregator_1 = Aggregate_LLM(
    {
        Prompt_Basic(),
        Prompt_Basic(),
        Prompt_Basic()
    }, Prompt_Basic(), aggregator_prompt
)

aggregator_2 = Aggregate_LLM(
    {
        Prompt_Basic(),
        Prompt_Basic(),
        Prompt_Basic()
    }, Prompt_Basic(), aggregator_prompt
)

aggregator_3 = Aggregate_LLM(
    {
        Prompt_Basic(),
        Prompt_Basic(),
        Prompt_Basic()
    }, Prompt_Basic(), aggregator_prompt
)

critique_prompt = get_critique_prompt()
rewrite_prompt = get_rewrite_prompt()

critique_1 = Critique_LLM(
    aggregator_1,
    Prompt_Basic(),
    Prompt_Basic(),
    critique_prompt,
    rewrite_prompt
)

critique_2 = Critique_LLM(
    aggregator_2,
    Prompt_Basic(),
    Prompt_Basic(),
    critique_prompt,
    rewrite_prompt
)

critique_3 = Critique_LLM(
    aggregator_3,
    Prompt_Basic(),
    Prompt_Basic(),
    critique_prompt,
    rewrite_prompt
)

final_aggregator = Aggregate_LLM(
    {
        critique_1,
        critique_2,
        critique_3
    }, Prompt_Basic(), aggregator_prompt
)

# Define query
query = generate_basic_prompt("What are the most important facts about George Washington?")

resp = responder.execute(model, final_aggregator, query)
print("Verbose Output:", resp.verbose_output)
print("Output:", resp.output)
