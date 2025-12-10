from openai import OpenAI, APIConnectionError, APITimeoutError
from inference_auth_token import get_access_token
import pandas as pd
import re
import sys

model_name = "google/gemma-3-27b-it"
#model_name = "meta-llama/Meta-Llama-3.1-70B-Instruct"

client = OpenAI(
    api_key=get_access_token(),
    base_url="https://inference-api.alcf.anl.gov/resource_server/sophia/vllm/v1"
)

# Emotion recognition categories
category_text = """Categories:
0: Q1 (happy   - high valence, high arousal)
1: Q2 (angry   - low  valence, high arousal)
2: Q3 (sad     - low  valence, low  arousal)
3: Q4 (relaxed - high valence, low  arousal)
"""

def call_llm(prompt, max_tokens=128, temperature=0.0):
    """Helper function to call LLM."""
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error calling LLM: {e}")
        return ""

def extract_abc_from_prompt(user_prompt):
    """Extract ABC score from user prompt."""
    # Try to extract ABC from triple backticks
    abc_match = re.search(r"```(.*?)```", user_prompt, re.S)
    if abc_match:
        return abc_match.group(1).strip()
    
    # Try "Input:" pattern - extract everything from "Input:" to "Task:" or "Options:"
    m1 = re.search(r"Input:\s*\n(.*?)(?=\n+Task:|\n+Options:)", user_prompt, re.S)
    if m1:
        result = m1.group(1).strip()
        if result:
            return result
    
    # Try "Score:" pattern
    m2 = re.search(r"Score:\s*\n(.*?)(?=\n+Task:|\n+Options:)", user_prompt, re.S)
    if m2:
        result = m2.group(1).strip()
        if result:
            return result
    
    # Fallback: try to find ABC notation patterns
    lines = user_prompt.split('\n')
    abc_lines = []
    in_abc = False
    for line in lines:
        stripped = line.strip()
        if not stripped and not in_abc:
            continue
        if re.match(r'^[XKMLR]:', stripped) or (stripped and re.search(r'[A-Ga-g][#b]?\d+', stripped)):
            abc_lines.append(stripped)
            in_abc = True
        elif in_abc:
            if stripped.startswith(('Task:', 'Options:', 'Answer:')):
                break
            elif stripped and not any(keyword in stripped for keyword in ['Input:', 'Task:', 'Options:', 'Answer:']):
                abc_lines.append(stripped)
            elif not stripped:
                if abc_lines:
                    abc_lines.append(stripped)
    
    if abc_lines:
        result = '\n'.join(abc_lines).strip()
        if result:
            return result
    
    return user_prompt.strip()

def build_emotion_classifier_prompt(abc_score):
    """Build prompt for direct emotion classification."""
    prompt = f"""{category_text}

{abc_score}

Output format: ONE number (0/1/2/3)

Answer:"""
    return prompt

def classify_emotion(user_prompt, temperature=0.0):
    """
    Direct emotion classification using single LLM call.
    Returns: (predicted_label, full_response)
    """
    # Extract ABC score from prompt
    abc_score = extract_abc_from_prompt(user_prompt)
    
    if not abc_score or len(abc_score) < 10 or not any(c in abc_score for c in ['X:', 'K:', 'M:', 'L:']):
        return ("", "Error: Could not extract ABC score")
    
    # Build prompt and call LLM
    prompt = build_emotion_classifier_prompt(abc_score)
    response = call_llm(prompt, max_tokens=4, temperature=temperature)
    
    # Extract label (0-3)
    label_match = re.search(r'\b([0-3])\b', response)
    if label_match:
        predicted_label = label_match.group(1)
    else:
        predicted_label = ""
    
    return (predicted_label, response)

# Main program entry point
if __name__ == "__main__":
    import sys
    
    # Check if running in batch mode (with CSV file) or interactive mode
    if len(sys.argv) > 1:
        # Batch mode: process CSV file
        csv_path = sys.argv[1]
        print(f"Processing CSV file: {csv_path}")
        
        df = pd.read_csv(csv_path)
        
        # Check if this is the rough4q format (has 'data' and 'label' columns)
        is_rough4q_format = "data" in df.columns and "label" in df.columns
        
        if is_rough4q_format:
            print("Detected rough4q_full_raw.csv format")
            # Build prompt from data column (ABC score)
            def build_prompt_from_data(row):
                abc_data = str(row.get("data", "")).strip()
                if not abc_data:
                    return ""
                # Build standard prompt format
                prompt = f"""Input:
{abc_data}

Task:
Choose the most probable emotional label of the provided score. Label Q1 refers to happy (high valence high arousal), Q2 refers to angry (low valence high arousal), Q3 refers to sad (low valence low arousal) and Q4 refers to relaxed (high valence low arousal).

Options:
0. Q1      1. Q2
2. Q3      3. Q4

Answer:"""
                return prompt
            
            df["prompt"] = df.apply(build_prompt_from_data, axis=1)
            # Use 'label' column as ground truth (instead of 'solution')
            df["solution"] = df["label"].astype(str)
            print(f"Total samples: {len(df)}")
            
            # Optional: filter by split (train/test)
            if len(sys.argv) > 3 and sys.argv[3] in ["train", "test", "val"]:
                split_filter = sys.argv[3]
                df = df[df["split"] == split_filter]
                print(f"Filtered to '{split_filter}' split: {len(df)} samples")
        else:
            # Original format (Emotion_Recognition_cleaned.csv)
            if "prompt" not in df.columns:
                print("Error: CSV file must have a 'prompt' column (or 'data' column for rough4q format)")
                sys.exit(1)
            print("Using original format (prompt column)")
        
        # Optional: limit number of samples for testing (random sampling)
        if len(sys.argv) > 2 and sys.argv[2].startswith('--'):
            n = int(sys.argv[2][2:])
            n = min(n, len(df))  # Ensure n doesn't exceed total samples
            df = df.sample(n=n, random_state=42).reset_index(drop=True)
            print(f"Randomly sampling {n} samples for testing")
        
        # Optional: temperature parameter
        temperature = 0.0
        if len(sys.argv) > 3 and sys.argv[3].startswith('--temp='):
            temperature = float(sys.argv[3].split('=')[1])
            print(f"Using temperature: {temperature}")
        
        results = []
        predicted_labels = []
        correct = 0
        
        for i, row in df.iterrows():
            print(f"\n{'='*60}")
            print(f"Processing sample {i+1}/{len(df)}...")
            user_prompt = row["prompt"]
            
            try:
                pred_label, full_response = classify_emotion(user_prompt, temperature=temperature)
                predicted_labels.append(pred_label)
                results.append(full_response)
                
                # Compare with ground truth (support both 'solution' and 'label' columns)
                gt_label = str(row.get("solution", row.get("label", "")))
                if pred_label == gt_label:
                    correct += 1
                    status = "✓ CORRECT"
                else:
                    status = "✗ WRONG"
                
                label_name = row.get("label_name", f"Q{int(gt_label)+1 if gt_label.isdigit() else '?'}")
                print(f"\n{status} | GT={gt_label} ({label_name}) | Pred={pred_label}")
                print(f"Response: {full_response[:100]}...")
                
            except Exception as e:
                print(f"Error processing sample {i+1}: {e}")
                results.append(f"Error: {str(e)}")
                predicted_labels.append("")
        
        # Save results (create a copy to avoid modifying original df)
        results_df = df.copy()
        results_df["agent_answer"] = results
        results_df["predicted_label"] = predicted_labels
        
        output_path = csv_path.replace(".csv", "_baseline_results.csv")
        results_df.to_csv(output_path, index=False)
        
        # Calculate accuracy
        accuracy = correct / len(df) if len(df) > 0 else 0
        print(f"\n{'='*60}")
        print(f"Results saved to: {output_path}")
        print(f"Accuracy: {correct}/{len(df)} = {accuracy:.4f}")
        print(f"Temperature: {temperature}")
        print(f"Model: {model_name}")
        
    else:
        # Interactive mode
        print("=" * 60)
        print("Emotion Classification Baseline (Direct LLM)")
        print("=" * 60)
        print("\nEnter your question about emotion classification.")
        print("Type 'quit' or 'exit' to stop.\n")
        
        while True:
            user_input = input("You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if not user_input:
                continue
            
            try:
                pred_label, full_response = classify_emotion(user_input)
                emotion_map = {"0": "Q1 (happy)", "1": "Q2 (angry)", "2": "Q3 (sad)", "3": "Q4 (relaxed)"}
                emotion_name = emotion_map.get(pred_label, "Unknown")
                print(f"\nPredicted: {emotion_name} (Label: {pred_label})")
                print(f"Full response: {full_response}\n")
            except Exception as e:
                print(f"\nError: {e}\n")

