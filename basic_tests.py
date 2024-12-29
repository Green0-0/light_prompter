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

# Define query
query = generate_basic_prompt("What is 10*2+156/3*2^3-sqrt(16*16)?", reread=True, planning=True, COT=True, answer_format="After reasoning, output **Answer:** in exactly that format with the asterisks, followed by the answer.")

print("=== Testing Basic Responders ===")
# Test Prompt_Basic
print("\n### Prompt_Basic")
r1 = Prompt_Basic()
resp1 = responder.execute(model, r1, query)
print("Verbose Output:", resp1.verbose_output)
if resp1.verbose_output != resp1.output:
    print("Error: Verbose output does not match output; output: ", resp1.output)
if resp1.verbose_output != resp1.answer:
    print("Error: Answer does not match output; answer:", resp1.answer)

# Test Prompt_Basic with answer extractor
print("\n### Prompt_Basic with Split Answer Extractor")
r2 = Prompt_Basic(answer_extractor=split_answer_extractor)
resp2 = responder.execute(model, r2, query)
print("Verbose Output:", resp2.verbose_output)
if resp2.verbose_output != resp2.output:
    print("Error: Verbose output does not match output; output: ", resp2.output)
print("Answer:", resp2.answer)

# Test Prompt_Basic with regex answer extractor
print("\n### Prompt_Basic with Regex Answer Extractor")
r3 = Prompt_Basic(answer_extractor=regex_answer_extractor)
query = generate_basic_prompt("What is 10*2+156/3*2^3-sqrt(16*16)?", reread=True, planning=True, COT=True, answer_format="After reasoning, output your final answer in \\boxed{}.")
resp3 = responder.execute(model, r3, query)
print("Verbose Output:", resp3.verbose_output)
if resp3.verbose_output != resp3.output:
    print("Error: Verbose output does not match output; output: ", resp3.output)
print("Answer:", resp3.answer)

print("\n=== Testing Two Turn Responders ===")
# Test Prompt_TwoTurn
print("\n### Prompt_TwoTurn")
r4 = Prompt_TwoTurn()
resp4 = responder.execute(model, r4, query)
print("Verbose Output:", resp4.verbose_output)

print("\n=== Testing Answer Selection Strategies ===")
query = "What is 10*2+156/3*2^3-sqrt(16*16)? Do not output anything besides the final answer. Do not add any formatting or thinking. Output only a single number."
# Test PickCommon_Exact
print("\n### PickCommon_Exact:")
picker_exact = PickCommon_Custom(
    {
        Prompt_Basic(), 
        Prompt_Basic(), 
        Prompt_Basic(), 
        Prompt_Basic(), 
        Prompt_Basic(),
        Prompt_Basic(), 
        Prompt_Basic(), 
        Prompt_Basic(), 
        Prompt_Basic(), 
        Prompt_Basic()
    })
resp_exact = responder.execute(model, picker_exact, query)
print("Verbose Output:", resp_exact.verbose_output)
print("Output:", resp_exact.output)
print("Answer:", resp_exact.answer)

# Test PickCommon_Custom with similarity function
print("\n### PickCommon_Custom:")
def numeric_similarity(a, b):
    try:
        a_num = float(a)
        b_num = float(b)
        if a_num == b_num:
            return 0
        return -abs(a_num - b_num)  # Negative distance for sorting
    except ValueError:
        return -float('inf')  # Treat non-numeric as least similar
        
picker_custom = PickCommon_Custom(
    {
        Prompt_Basic(), 
        Prompt_Basic(), 
        Prompt_Basic(), 
        Prompt_Basic(), 
        Prompt_Basic(),
        Prompt_Basic(), 
        Prompt_Basic(), 
        Prompt_Basic(), 
        Prompt_Basic(), 
        Prompt_Basic()
    }, numeric_similarity)
resp_custom = responder.execute(model, picker_custom, query)
print("Verbose Output:", resp_custom.verbose_output)
print("Output:", resp_custom.output)
print("Answer:", resp_custom.answer)

print("\n=== Testing Aggregation and Critique ===")
query = generate_basic_prompt("Find the number of cubic polynomials $p(x) = x^3 + ax^2 + bx + c,$ where $a, b,$ and $c$ are integers in $\{-20,-19,-18,\ldots,18,19,20\},$ such that there is a unique integer $m \not= 2$ with $p(m) = p(2).$", reread=True, planning=True, COT=True, answer_format="After reasoning, output your final answer in \\boxed{}.")

aggregation_format = get_grading_prompt()

# Test grading-based aggregation
print("\n### Grading Based Aggregation:")
aggregator = Aggregate_LLM(
    {
        Prompt_Basic(answer_extractor=regex_answer_extractor), 
        Prompt_Basic(answer_extractor=regex_answer_extractor), 
        Prompt_Basic(answer_extractor=regex_answer_extractor)
    },
    Prompt_Basic(answer_extractor=split_answer_extractor),
    aggregation_format,
    final_answer_extractor=regex_answer_extractor
)
resp_agg = responder.execute(model, aggregator, query)
print("Verbose Output:", resp_agg.verbose_output)
print("Output:", resp_agg.output)
print("Answer:", resp_agg.answer)

# Test critique and refinement
print("\n### Critique and Refinement:")
critique_prompt = get_critique_prompt()
rewrite_prompt = get_rewrite_prompt()
critic = Critique_LLM(
    Prompt_Basic(answer_extractor=regex_answer_extractor),
    Prompt_Basic(answer_extractor=regex_answer_extractor),
    Prompt_Basic(answer_extractor=regex_answer_extractor),
    critique_prompt,
    rewrite_prompt
)
resp_crit = responder.execute(model, critic, query)
print("Verbose Output:", resp_crit.verbose_output)
print("Output:", resp_crit.output)
print("Answer:", resp_crit.answer)
