from openai import OpenAI, APIConnectionError, APITimeoutError
from inference_auth_token import get_access_token
import pandas as pd
import re
import sys
from collections import Counter

#model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
model_name = "google/gemma-3-27b-it"
df = pd.read_csv("data/Emotion_Recognition_cleaned.csv")

if len(sys.argv) > 1 and sys.argv[1].startswith('--'):
    n = int(sys.argv[1][2:])
    df = df.head(n)

client = OpenAI(
    api_key=get_access_token(),
    base_url="https://inference-api.alcf.anl.gov/resource_server/sophia/vllm/v1"
)

num_analysts = 3
num_fewshot = 6

category_text = """Categories:
0: Q1 (happy   - high valence, high arousal)
1: Q2 (angry   - low  valence, high arousal)
2: Q3 (sad     - low  valence, low  arousal)
3: Q4 (relaxed - high valence, low  arousal)
"""

def build_fewshot(df, k=num_fewshot):
    k = min(k, len(df))
    rows = df.sample(k, random_state=42)
    examples = []
    for idx, r in rows.iterrows():
        ex = f"""Example:
Score:
{r['prompt']}

Correct label: {r['solution']}"""
        examples.append(ex)
    return "\n\n".join(examples)

fewshot_block = build_fewshot(df, num_fewshot)

def call_llm(prompt, max_tokens=8, temperature=0.0):
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
            stop=["\n"]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error calling LLM: {e}")
        return ""


# CHANGED: 要求 analyst 输出 LABEL + REASON
analyst_instruction = f"""You are an emotion classifier for musical scores written in ABC notation.

Your job:
- Read the input score.
- Decide which ONE of the following 4 categories it belongs to.
- THEN briefly explain why.
- IMPORTANT: You MUST follow this exact output format:

Line 1: LABEL: <one number 0, 1, 2, or 3>
Line 2: REASON: <your short explanation in 1-3 sentences>

{category_text}

Here are some examples of how scores are labeled:
{fewshot_block}
"""

def build_analyst_prompt(prompt):
    return f"""{analyst_instruction}

Now classify the following score:

Score:
{prompt}

Remember: follow the LABEL / REASON format exactly.
"""

def analyst_answer_once(prompt):
    analyst_prompt = build_analyst_prompt(prompt)
    # CHANGED: 解释会长一点，给多点 token
    # Use higher temperature (0.7) to encourage diversity in analyst opinions for voting
    return call_llm(analyst_prompt, max_tokens=128, temperature=0.7)


# CHANGED: prompt 里提到 LABEL 格式，逻辑不变
format_checker_instruction = f"""You are a strict format checker.

You are given an answer that should look like:

LABEL: <one number 0, 1, 2, or 3>
REASON: <some explanation>

{category_text}

Your task:
- Extract the label index (0, 1, 2, or 3) from the given answer.
- If there is more than one number, use the one that appears after 'LABEL:'.
- If you cannot find a valid number (0-3) anywhere, respond with "INVALID".

Your response must be EITHER:
- A single digit 0/1/2/3
- Or the word INVALID"""

def format_checker_llm(original_answer):
    check_prompt = f"""{format_checker_instruction}

Original answer:
{original_answer}

Your response (ONLY 0/1/2/3 or INVALID):"""
    return call_llm(check_prompt, max_tokens=4, temperature=0.0)


def extract_option_index(text, num_options=4):
    if not text:
        return ""
    m = re.search(r'\b([0-3])\b', str(text))
    if m:
        idx = m.group(1)
        if int(idx) < num_options:
            return idx
    return ""

# NEW: 从 analyst 的回答中抽取 REASON 文本
def extract_reason(text: str) -> str:
    if not text:
        return ""
    lines = str(text).splitlines()
    for line in lines:
        if line.strip().upper().startswith("REASON:"):
            return line.split(":", 1)[1].strip()
    # 如果没有 REASON: 行，就把整段当理由兜底
    return str(text).strip()


# CHANGED: Judge 现在会看 label + reason
judge_instruction = f"""You are a meta-judge combining the opinions of several analyst agents.

Each analyst has independently classified the same musical score into one of 4 categories and provided a short explanation.

Your job:
- Look at the original score.
- Look at all analyst predictions AND their reasons.
- Consider how reasonable each explanation is given the 4 emotion definitions.
- Also consider how many analysts voted for each label.
- Decide the SINGLE best label (0, 1, 2, or 3).

{category_text}

Guidelines:
- If there is a clear majority AND their reasoning matches the category definition, you should usually follow the majority.
- If analysts are split, prefer the label whose explanations best match the musical features in the score and the category definitions.
- If some explanations are clearly wrong or inconsistent with the score or the definitions, you may ignore those votes.
- You MUST output ONLY the final label number (0/1/2/3). Do NOT explain."""

# CHANGED: 多传一个 analyst_reasons
def build_judge_prompt(prompt, analyst_answers, clean_labels, analyst_reasons):
    lines = []
    for i, (lab, reason) in enumerate(zip(clean_labels, analyst_reasons), start=1):
        lines.append(f"Analyst {i}:\n  Label: {lab}\n  Reason: {reason}")
    summary = "\n\n".join(lines)

    counts = Counter([l for l in clean_labels if l in ["0", "1", "2", "3"]])
    count_str = ", ".join(f"{k}: {v}" for k, v in sorted(counts.items()))

    return f"""{judge_instruction}

Score:
{prompt}

Analyst predictions (label + reason):
{summary}

Vote counts by label: {count_str if count_str else "no valid labels"}

Now, choose the SINGLE best label for this score.

Your answer (ONLY one number 0/1/2/3):"""

def content_checker_llm(prompt, analyst_answers, clean_labels, analyst_reasons):
    judge_prompt = build_judge_prompt(prompt, analyst_answers, clean_labels, analyst_reasons)
    return call_llm(judge_prompt, max_tokens=4, temperature=0.0)


predictions_single = []
predictions_majority = []
predictions_agent = []
raw_analyst_answers = []
raw_format_checks = []
raw_judge_answers = []
analyst_reasons_all = []   # NEW: 保存所有样本的 reasons

correct_single = 0
correct_majority = 0
correct_agent = 0

for i, row in df.iterrows():

    prompt = row["prompt"]

    try:
        analyst_answers = []
        clean_labels = []
        analyst_reasons = []  # NEW: 当前样本的 reasons
        format_checks = []  # 当前样本的 format checks

        for k in range(num_analysts):
            ans = analyst_answer_once(prompt)
            analyst_answers.append(ans)

            fmt = format_checker_llm(ans)
            format_checks.append(fmt)

            if fmt.strip().upper() == "INVALID":
                lab = extract_option_index(ans)
            else:
                lab = extract_option_index(fmt)
            clean_labels.append(lab)

            # NEW: 提取理由
            reason = extract_reason(ans)
            analyst_reasons.append(reason)

        raw_analyst_answers.append(analyst_answers)
        analyst_reasons_all.append(analyst_reasons)

        single_label = clean_labels[0] if clean_labels and clean_labels[0] in ["0", "1", "2", "3"] else ""
        predictions_single.append(single_label)

        if str(single_label) == str(row["solution"]):
            correct_single += 1

        valid_labels = [l for l in clean_labels if l in ["0", "1", "2", "3"]]
        if valid_labels:
            majority_label = Counter(valid_labels).most_common(1)[0][0]
        else:
            majority_label = ""
        predictions_majority.append(majority_label)

        if str(majority_label) == str(row["solution"]):
            correct_majority += 1

        # CHANGED: 传入 analyst_reasons
        judge_answer = content_checker_llm(prompt, analyst_answers, clean_labels, analyst_reasons)
        raw_judge_answers.append(judge_answer)

        final_label = extract_option_index(judge_answer)
        if not final_label and valid_labels:
            final_label = majority_label

        predictions_agent.append(final_label)

        if str(final_label) == str(row["solution"]):
            correct_agent += 1

        print(f"[{i}] GT={row['solution']} | single={single_label} | majority={majority_label} | agent={final_label}")

    except Exception as e:
        print("Error at sample", i, e)
        predictions_single.append("")
        predictions_majority.append("")
        predictions_agent.append("")
        raw_analyst_answers.append([])
        raw_format_checks.append([])
        raw_judge_answers.append("")
        analyst_reasons_all.append([])

results_df = pd.DataFrame({
    'index': df.index,
    'ground_truth': df['solution'].values,
    'prediction_single': predictions_single,
    'prediction_majority': predictions_majority,
    'prediction_agent': predictions_agent,
    'raw_analyst_answers': raw_analyst_answers,
    'raw_format_checks': raw_format_checks,
    'analyst_reasons': analyst_reasons_all,     # NEW: 存每个样本的 reason 列表
    'raw_judge_answer': raw_judge_answers
})
results_df.to_csv('emotion_recognition_agent_results.csv', index=False)

accuracy_single = correct_single / len(df)
accuracy_majority = correct_majority / len(df)
accuracy_agent = correct_agent / len(df)

print("\n===========================")
print(f"Model: {model_name}")
print(f"Single LLM accuracy: {accuracy_single:.4f}")
print(f"Majority vote accuracy: {accuracy_majority:.4f}")
print(f"Multi-agent accuracy: {accuracy_agent:.4f}")
