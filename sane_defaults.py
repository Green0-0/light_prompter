def split_answer_extractor(text: str, splitter: str = "**Answer:**") -> str:
    """Splits the text by splitter and returns the last part"""
    parts = text.split(splitter)
    return parts[-1].strip()

def regex_answer_extractor(text: str, regex_key: str = r"\\boxed{(.*?)}") -> str:
    """Finds matches using regex and returns the last match"""
    import re
    matches = re.findall(regex_key, text)
    return matches[-1] if matches else text

def generate_basic_prompt(query : str, reread = False, planning = False, COT = False, answer_format = ""):
    """
    Generates a prompt string based on the given parameters.

    Args:
        reread (bool): If True, instructs to re-read and reflect on the query and context.
        planning (bool): If True, instructs to generate a detailed plan to complete the query.
        COT (bool): If True, instructs to think step by step in a detailed manner.
        answer_format (str): The format to answer in.

    Returns:
        str: A formatted prompt string based on the provided arguments.
    """

    prompt = f"""**Query:** 
    {query}
    
    **Instruction:**
    Answer the above query."""
    start = " Begin by "
    if reread:
        prompt += start + "re-reading the query, and reflect upon any important details within the query and context that will assist you in answering it."
        start = " Then begin "
    if planning:
        prompt += start + "generating a detailed plan to complete the query, breaking it down into small steps you will take to reach an answer."
        if reread:
            start = " Finally "
        else:
            start = " Then begin "
    if COT:
        prompt += start + "thinking step by step to answer the query. Be detailed in each step, describing every operation you do."
    prompt += answer_format
    return prompt

def get_aggregate_prompt():
    """
    Returns a prompt string for aggregating responses.
    """
    
    prompt = """
    **Original query:**
    {{query}}
    ------------
    **Here is a list of responses to that query:**
    {{responses}}
    -------------
    Your task is to synthesize these responses into a single, high-quality response. It is crucial to critically evaluate the information provided in these responses, recognizing that some of it may be biased or incorrect. Your response should not simply replicate the given answers but should offer a refined, accurate, and comprehensive reply to the query. Ensure your response is well-structured, coherent, and adheres to the highest standards of accuracy and reliability."""
    return prompt

def get_grading_prompt(criteria = ["helpfulness", "accuracy", "formatting", "clarity"]):
    """
    Returns a prompt string for grading responses and picking the best one.
    """
    prompt = """
    **Original query:**
    {{query}}
    ------------
    **Here is a list of responses to that query:**
    {{responses}}
    -------------
    For each of the responses, first analyze how it responds to the query and what it says. Then reflect upon each of the following criteria: {{criteria}}, and decide how well the response meets these criteria. Be as critical as possible, nitpicking details and things the response could have done better.
    After analyzing and grading each response, output \"**Answer:**\" followed by a word for word copy of the best response, making sure to not leave out anything. 
    """.replace("{{criteria}}", ", ".join(criteria))
    return prompt

def get_critique_prompt(criteria = ["helpfulness", "accuracy", "formatting", "clarity"]):
    """
    Returns a prompt string for generating critiques.
    """
    prompt = """
    **Original query:**
    {{query}}
    ------------
    **Here is a response to that query:**
    {{response}}
    -------------
    Your task is to criticize the response based on the following criteria: {{criteria}}.
    First analyze how the response responds to the query and what it says. Then reflect upon each of the following criteria: {{criteria}}, and decide how well the response meets these criteria. Offer detailed suggestions for how to improve the response, with the goal of creating the best response possible. Do not attempt to rewrite the response yourself, your job is to simply give criticism.
    """.replace("{{criteria}}", ", ".join(criteria))
    return prompt

def get_rewrite_prompt():
    """
    Returns a prompt string for rewriting responses based on critique.
    """
    prompt = """
    **Original query:**
    {{query}}
    ------------
    **Here is a response to that query:**
    {{responses}}
    -------------
    Your task is to rewrite and improve the response based on the following critique: {{critique}}.
    """
    return prompt