from openai import OpenAI, APIConnectionError, APITimeoutError
from inference_auth_token import get_access_token
import pandas as pd
import re
import sys

#model_name = "google/gemma-3-27b-it"
model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
df = pd.read_csv("data/Bar_Count_Estimation.csv")

if len(sys.argv) > 1 and sys.argv[1].startswith('--'):
    n = int(sys.argv[1][2:])
    df = df.head(n)

client = OpenAI(
    api_key=get_access_token(),
    base_url="https://inference-api.alcf.anl.gov/resource_server/sophia/vllm/v1"
)


def extract_bar_count(pred_raw):
    """
    Extract the bar count number from the model's response.
    The model should output a number representing the number of bars.
    """
    if pred_raw is None:
        return ""

    text = pred_raw.strip()

    # Try to find a number in the response
    # Look for patterns like "8", "8 bars", "The answer is 8", etc.
    
    # 1) Look for "X bars" or "X bar" format
    m = re.search(r"\b(\d+)\s+bar", text, re.IGNORECASE)
    if m:
        return m.group(1)
    
    # 2) Look for "answer is X" or "is X" format
    m = re.search(r"(?:answer|count|number|bars?)\s+(?:is|are|:)?\s*(\d+)", text, re.IGNORECASE)
    if m:
        return m.group(1)
    
    # 3) Look for standalone number (prefer numbers that are reasonable for bar counts, e.g., 1-100)
    m = re.search(r"\b([1-9]\d?|100)\b", text)
    if m:
        num = int(m.group(1))
        if 1 <= num <= 100:  # Reasonable range for bar counts
            return str(num)
    
    # 4) Fallback: any number
    m = re.search(r"\b(\d+)\b", text)
    if m:
        return m.group(1)
    
    # 5) No number found
    return ""


predictions = []
raw_responses = []
correct = 0

for i, row in df.iterrows():
    
    # Build prompt similar to Metadata_QA format
    input_content = row["score"]
    task = row["task_description"]
    
    prompt = f"""Input:
{input_content}

Task:
{task}

Answer:"""

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )

        raw_response = response.choices[0].message.content
        pred = extract_bar_count(raw_response)
        raw_responses.append(raw_response)

    except Exception as e:
        print("Error at sample", i, e)
        pred = ""
        raw_responses.append("")

    predictions.append(pred)

    # accuracy
    if str(pred) == str(row["target"]):
        correct += 1

    print(f"[{i}] GT={row['target']} | Pred={pred}")

# Save results to CSV
results_df = pd.DataFrame({
    'index': df.index,
    'ground_truth': df['target'].values,
    'prediction': predictions,
    'raw_response': raw_responses
})
results_df.to_csv('bar_count_results.csv', index=False)


accuracy = correct / len(df)

print("\n===========================")
print(f"Model: {model_name}")
print(f"Accuracy: {accuracy:.4f}")

