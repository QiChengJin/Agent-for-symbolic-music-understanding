import pandas as pd
from pathlib import Path
# path of data
data_dir = Path("data")    


# ==============================
# 1. Error Detection
# ==============================


error_detection_csv = data_dir / "error_detection.csv"
ed_df = pd.read_csv(error_detection_csv)

def parse_error_list(err_string):
    return [e.strip() for e in err_string.split(",")]
    
ed_df["error_list"] = ed_df["error"].apply(parse_error_list)

def build_prompt_ed(row):
    input_content = row["input"]
    task = row["task_description"]
    options = row["error_list"]

    # build the options text block
    options_text = "\n".join([f"{i}. {opt}" for i, opt in enumerate(options)])

    prompt = f"""Input:
{input_content}

Task:
{task}

Options:
{options_text}
"""
    return prompt

ed_df["prompt"] = ed_df.apply(build_prompt_ed, axis=1)


ed_clean_df = ed_df.drop(columns = ["title","input","choices","target","task_description","error", "error_list"])
ed_clean_df = ed_clean_df.rename(columns={"target_index": "solution"})

output_path = data_dir / "Error_Detection_cleaned.csv"
ed_clean_df.to_csv(output_path, index=False)




# ==============================
# 2. Metadata_QA
# ==============================


metadata_qa = data_dir / "Metadata_QA.csv"
qa_df = pd.read_csv(metadata_qa)


import ast

def build_prompt_qa(row):
    input_content = row["score"]
    task = row["task_description"]

    # Convert string list "['Dm','D','G','E']" → real Python list
    choices = ast.literal_eval(row["choices"])

    # Format as:
    # 0. A 1. B
    # 2. C 3. D
    options_text = (
        f"0. {choices[0]}   1. {choices[1]}\n"
        f"2. {choices[2]}   3. {choices[3]}"
    )

    prompt = f"""Input:
{input_content}

Task:
{task}

Options:
{options_text}

Answer:"""

    return prompt


qa_df["prompt"] = qa_df.apply(build_prompt_qa, axis=1)


qa_clean_df = qa_df.drop(columns = ["title","score", "choices", "target","task_description"])
qa_clean_df = qa_clean_df.rename(columns={"target_index": "solution"})
output_path = data_dir / "Metadata_QA_cleaned.csv"
qa_clean_df.to_csv(output_path, index=False)




# ==============================
# 2. Emotion_Recognition 
# ==============================


metadata_er = data_dir / "Emotion_Recognition.csv"
er_df = pd.read_csv(metadata_er)


import ast

def build_prompt_er(row):
    input_content = row["score"]
    task = row["task_description"]


    prompt = f"""Input:
{input_content}

Task:
{task}

Options:
0. Q1      1. Q2
2. Q3      3. Q4

Answer:"""

    return prompt


er_df["prompt"] = er_df.apply(build_prompt_er, axis=1)


er_clean_df = er_df.drop(columns = ["title","score", "choices", "target","task_description"])
er_clean_df = er_clean_df.rename(columns={"target_index": "solution"})
output_path = data_dir / "Emotion_Recognition_cleaned.csv"
er_clean_df.to_csv(output_path, index=False)


















