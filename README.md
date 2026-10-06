# M.U.S.E. — Multi-Agent Symbolic Music Understanding

M.U.S.E. is an LLM-based multi-agent system for reasoning over music written in
[ABC notation](https://abcnotation.com/). It routes each question to specialized
agents for score analysis, emotion recognition, or both, then evaluates and
combines their answers.

The project explores a practical question: **can role-specialized LLM agents
reason about symbolic music more reliably than a single direct prompt?**

> Research prototype developed with the Argonne Leadership Computing Facility
> (ALCF) inference service. Running the models requires an eligible Globus
> identity; the code, datasets, prompts, paper, and recorded results are included
> for inspection.

## What it does

- Analyzes key, meter, harmony, rhythm, structure, and other ABC-score metadata.
- Classifies emotion on a valence–arousal plane: happy, angry, sad, or relaxed.
- Validates and extracts ABC notation before inference.
- Routes technical, emotional, and mixed questions to dedicated agents.
- Uses multiple arousal/valence analysts and a judge to reduce single-prompt
  variance.
- Supports interactive use and batch evaluation from CSV files.

## Architecture

```text
User prompt
    │
    ▼
Input validator ── extracts and checks ABC notation
    │
    ▼
Controller ─────── classifies the request as ABC / EMOTION / BOTH
    │
    ├── ABC expert ── evaluator
    │
    └── Emotion team ── 3× arousal + 3× valence analysts ── judge
    │
    ▼
Response aggregator
```

The main implementation is in
[`src/music_agent/system.py`](src/music_agent/system.py). A detailed design and
prompt walkthrough is available in
[`docs/SYSTEM_DOCUMENTATION.md`](docs/SYSTEM_DOCUMENTATION.md).

## Recorded results

The following values are recalculated from the checked-in CSV outputs. They are
small research runs rather than benchmark claims.

| Task / configuration | Samples | Accuracy |
|---|---:|---:|
| Emotion, direct prediction | 50 | 26.0% |
| Emotion, single analyst | 50 | 44.0% |
| Emotion, majority vote | 50 | 46.0% |
| Emotion, agent judge | 50 | **50.0%** |
| Metadata QA, Llama baseline | 60 | 25.0% |
| Metadata QA, Gemma baseline | 60 | **98.3%** |

Reproduce the table locally with:

```bash
python scripts/summarize_results.py
```

The repository contains 619 task examples across bar counting, bar sequencing,
error detection, metadata QA, and emotion recognition. Raw and prompt-ready
versions are separated under `data/`.

## Quick start

Requirements: Python 3.10+ and an ALCF-authorized Globus account.

```bash
git clone https://github.com/QiChengJin/Agent-for-symbolic-music-understanding.git
cd Agent-for-symbolic-music-understanding

python -m venv .venv
source .venv/bin/activate
pip install -e .

# First-time browser authentication
python -m music_agent.auth authenticate

# Interactive agent
python -m music_agent.system
```

Example input:

```text
Input:
X:1
T:Example
M:4/4
L:1/8
K:C
CDEF GABc |

Task: What is the key and which emotion does this melody most likely express?
```

For batch evaluation, pass a CSV containing a `prompt` column:

```bash
python -m music_agent.system data/processed/Metadata_QA_cleaned.csv
```

## Repository layout

```text
.
├── src/music_agent/       # Importable core system and Globus authentication
├── experiments/
│   ├── agents/            # Voting, reasoning, and metadata-agent variants
│   ├── baselines/         # Direct-prompt baselines
│   └── evaluation/        # Standalone emotion-system evaluation
├── data/
│   ├── raw/               # Original task datasets
│   └── processed/         # Prompt-ready datasets
├── results/               # Recorded experiment outputs
├── scripts/               # Data preparation and result summaries
└── docs/
    ├── SYSTEM_DOCUMENTATION.md
    └── paper/              # Project paper, LaTeX sources, and figures
```

## Experiments and reproducibility

All experiment scripts use paths relative to the repository root and write
outputs to `results/`. After `pip install -e .`, examples include:

```bash
python experiments/baselines/emotion_direct.py --10
python experiments/agents/emotion_reasoning.py --10
python experiments/agents/metadata_qa.py
python scripts/prepare_data.py
```

Model identifiers and inference parameters are intentionally kept next to each
experiment so recorded configurations are easy to audit. The current scripts
target ALCF's OpenAI-compatible inference endpoint.

## Project report

The full report, **“M.U.S.E. Multi-agent Symbolic Music Understanding with
LLMs,”** is available as a [PDF](docs/paper/MUSE-paper.pdf). Its LaTeX sources
and figures are included in [`docs/paper/source/`](docs/paper/source/).

## Notes

- No access tokens are stored in this repository. Globus stores local tokens
  outside the project under the user's home directory.
- Results can vary with model serving versions and sampling settings.
- This is a research prototype, not a production music-analysis service.
