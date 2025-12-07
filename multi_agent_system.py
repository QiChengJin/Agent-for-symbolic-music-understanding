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


def agent_B_abc_system(user_prompt):

    # Extract ABC from user prompt
    # Assumes ABC text appears after "Score:" or triple backticks
    abc_match = re.search(r"```(.*?)```", user_prompt, re.S)
    if abc_match:
        abc_text = abc_match.group(1)
    else:
        # fallback try "Score:"
        m2 = re.search(r"Score:(.*)", user_prompt, re.S)
        abc_text = m2.group(1).strip() if m2 else ""

    # Step 1: Expert analysis
    analysis = abc_expert_agent(abc_text)

    # Step 2: Evaluator answers user question
    answer = evaluator_agent(analysis, user_prompt)

    return answer

def agent_C_emotion_system(user_prompt):
    """
    TODO:
    """
    return "Agent C placeholder response (emotion analysis not implemented yet)"


def controller_prompt(user_prompt):
    return f"""
You are the Controller Agent.

Your job is to decide which specialized agents should be used.

Rules:
- If the user asks about ABC notation, music score structure, bars, keys -> respond "ABC"
- If the user asks about emotion, valence/arousal, mood detection -> respond "EMOTION"
- If both topics appear → respond "BOTH"
- If neither → respond "NONE"

Return ONLY ONE WORD: ABC, EMOTION, BOTH, or NONE.

User prompt:
{user_prompt}
"""


def agent_A_controller(user_prompt):
    decision = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": controller_prompt(user_prompt)}],
        temperature=0
    ).choices[0].message.content.strip()

    # Normalize
    decision = decision.upper()

    return decision

def agent_D_aggregator(answer_B=None, answer_C=None):
    text = ""

    if answer_B:
        text += f"🎼 **ABC Score Expert Answer:**\n{answer_B}\n\n"

    if answer_C:
        text += f"🎵 **Emotion Expert Answer:**\n{answer_C}\n\n"

    if not text:
        text = "No specialized agent was required. No additional information."

    return text


def run_agent_system(user_prompt):

    decision = agent_A_controller(user_prompt)
    print("Controller decision:", decision)

    answer_B = None
    answer_C = None

    if decision in ("ABC", "BOTH"):
        answer_B = agent_B_abc_system(user_prompt)

    if decision in ("EMOTION", "BOTH"):
        answer_C = agent_C_emotion_system(user_prompt)

    final_answer = agent_D_aggregator(answer_B, answer_C)
    return final_answer

