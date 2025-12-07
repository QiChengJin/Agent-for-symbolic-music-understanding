from openai import OpenAI, APIConnectionError, APITimeoutError
from inference_auth_token import get_access_token
import pandas as pd
import re
model_name = "google/gemma-3-27b-it"

client = OpenAI(
    api_key=get_access_token(),
    base_url="https://inference-api.alcf.anl.gov/resource_server/sophia/vllm/v1"
)




def abc_expert_prompt(input_abc):
    return f"""
You are an ABC notation expert. Your job is to interpret the following ABC score.

Score:
{input_abc}

Explain the meaning of each ABC component in a structured and concise way.
Focus on:
- Key (K:)
- Meter / time signature (M:)
- Default note length (L:)
- Chord symbols
- Bar boundaries
- Rhythm patterns
- Melodic contour
- Tuplets and ornaments
- Phrase structure

Do NOT answer the user's question.
ONLY produce an analysis of the ABC score.
"""

def evaluator_prompt(analysis, task_prompt):
    return f"""
You are the evaluator agent.

You will receive an analysis of an ABC score from the ABC Expert.
Your job is to answer the user's question based ONLY on that analysis.

ABC Expert Analysis:
{analysis}

Task:
{task_prompt}
"""

def abc_expert_agent(input_abc):
    prompt = abc_expert_prompt(input_abc)
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return response.choices[0].message.content


def evaluator_agent(analysis, full_prompt, num_options):
    prompt = evaluator_prompt(analysis, full_prompt)
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    raw = response.choices[0].message.content
    return raw

