# Multi-Agent System for Symbolic Music Understanding

## 📋 目录

- [系统概述](#系统概述)
- [架构设计](#架构设计)
- [Agent 详细说明](#agent-详细说明)
- [数据格式](#数据格式)
- [安装与配置](#安装与配置)
- [运行指南](#运行指南)
- [代码结构](#代码结构)
- [关键实现细节](#关键实现细节)
- [复现指南](#复现指南)
- [使用示例](#使用示例)

---

## 系统概述

这是一个基于多智能体（Multi-Agent）架构的符号音乐理解系统，专门设计用于处理 ABC 乐谱格式的音乐分析任务。系统能够处理两类主要任务：

1. **ABC 乐谱技术分析**：分析乐谱的结构、调性、节拍、和弦等技术特征
2. **音乐情感分类**：将音乐分类为四种情感类别（Q1-Q4），基于效价（valence）和唤醒度（arousal）维度

### 核心特性

- **输入验证**：在处理前验证 ABC 乐谱提取和问题存在，确保输入质量
- **智能路由**：使用 LLM 自动识别问题类型并路由到相应的专业智能体
- **多智能体协作**：不同智能体专注于不同任务，提高准确性
- **任务拆分**：当问题同时涉及多个领域时，自动拆分任务并分别处理
- **情感分类的二维模型**：基于 arousal-valence 二维模型，使用多智能体方法提高分类准确性

---

## 架构设计

### 整体架构图

```
                    User Prompt
                         |
                         v
              ┌──────────────────────┐
              │ Input Validator      │
              │   - LLM-based        │
              │   - ABC extraction   │
              │   - Question check  │
              └──────────┬───────────┘
                         |
                         v
              ┌──────────────────────┐
              │  Agent A (Controller)│
              │   - LLM-based        │
              │   - Decision Maker   │
              └──────────┬───────────┘
                         |
        ┌────────────────┼────────────────┐
        |                |                |
        v                v                v
    "ABC"           "EMOTION"          "BOTH"
        |                |                |
        |                |         ┌───────┴───────┐
        |                |         | Task Splitter |
        |                |         └───────┬───────┘
        |                |                 |
        v                v                 |
┌───────────────┐ ┌──────────────┐        |
│ Agent B       │ │ Agent C      │        |
│ (ABC Expert)  │ │ (Emotion)    │        |
│               │ │              │        |
│ 1. ABC Expert │ │ 1. Arousal   │        |
│ 2. Evaluator  │ │   Analysts(3) │        |
│               │ │ 2. Valence   │        |
│               │ │   Analysts(3)│        |
│               │ │ 3. Combiner  │        |
└───────┬───────┘ └──────┬───────┘        |
        |                |                 |
        └────────────────┴─────────────────┘
                         |
                         v
              ┌──────────────────────┐
              │ Agent D (Aggregator)  │
              │   - Combine Answers   │
              └──────────┬────────────┘
                         |
                    Final Answer
```

### Agent 角色说明

| Agent | 角色 | 功能 | LLM 调用次数 |
|-------|------|------|-------------|
| **Input Validator** | 输入验证器 | 验证 ABC 乐谱提取和问题存在 | 1 |
| **Agent A** | Controller | 分析用户输入，决定使用哪个智能体 | 1 |
| **Agent B** | ABC Expert System | 分析 ABC 乐谱并回答技术问题 | 2 (Expert + Evaluator) |
| **Agent C** | Emotion System | 分类音乐情感（基于 arousal-valence） | 7 (3 Arousal + 3 Valence + 1 Combiner) |
| **Agent D** | Aggregator | 聚合多个智能体的答案 | 0 (纯文本处理) |

---

## Agent 详细说明

### Input Validator（输入验证器）

**功能**：在 Controller 之前验证用户输入，确保输入包含有效的 ABC 乐谱和明确的问题。

**工作流程**：

1. **脚本提取 ABC**：首先使用正则表达式尝试从 prompt 中提取 ABC 乐谱
   - 尝试从代码块（```）中提取
   - 尝试从 "Input:" 或 "Score:" 后提取
   - 尝试识别 ABC 标记（X:, K:, M:, L: 等）

2. **LLM 验证**：
   - **情况 1：未提取到 ABC**
     - 调用 LLM 尝试从用户输入中提取 ABC 乐谱
     - 如果 LLM 确认没有 ABC，返回错误并要求用户重新输入
     - 如果 LLM 提取到 ABC，继续验证问题存在
   
   - **情况 2：已提取到 ABC**
     - 调用 LLM 验证 ABC 乐谱是否完整
     - 检查是否遗漏了某些部分
     - 检查用户是否提出了明确的问题

3. **验证结果处理**：
   - `VALID_INPUT`: 输入有效，继续处理
   - `NO_ABC_SCORE`: 未检测到 ABC 乐谱，返回错误信息
   - `INCOMPLETE_ABC`: ABC 乐谱不完整，提示用户补充缺失部分
   - `NO_QUESTION_DETECTED`: 未检测到问题，要求用户提问

**实现细节**：

```python
def validate_input(user_prompt):
    """
    验证用户输入
    Returns: (is_valid, error_message, verified_abc)
    """
    # Step 1: 脚本提取 ABC
    extracted_abc = extract_abc_from_prompt(user_prompt)
    has_abc = extracted_abc and len(extracted_abc) > 10 and 
              any(c in extracted_abc for c in ['X:', 'K:', 'M:', 'L:'])
    
    # Step 2: LLM 验证
    validation_prompt = input_validator_prompt(user_prompt, extracted_abc, has_abc)
    validation_result = call_llm(validation_prompt, temperature=0, max_tokens=300)
    
    # Step 3: 解析验证结果并返回
    ...
```

**Prompt 设计**：

当未检测到 ABC 时：
```python
"""You are an input validator for a music analysis system.

Your task is to check if the user input contains an ABC notation score.

User input:
{user_prompt}

ABC notation typically:
- Starts with headers like X:, K:, M:, L:, R:
- Contains musical notes (A-G with optional sharps/flats and octaves)
- May be wrapped in code blocks (```) or after "Input:" or "Score:"

Please analyze the user input and:
1. If you find ABC notation, extract it completely and respond with:
   EXTRACTED_ABC:
   [the complete ABC score here]

2. If you confirm there is NO ABC notation, respond with:
   NO_ABC_SCORE
"""
```

当已检测到 ABC 时：
```python
"""You are an input validator for a music analysis system.

Your task is to:
1. Verify that the extracted ABC score is complete and correct
2. Check if the user has asked a question

User input:
{user_prompt}

Extracted ABC score:
{extracted_abc}

Please analyze and respond in one of these formats:

If the ABC score is complete and correct, AND the user has asked a question:
VALID_INPUT

If the ABC score is incomplete or missing parts:
INCOMPLETE_ABC:
[description of what's missing or what should be added]

If the user has NOT asked a question:
NO_QUESTION_DETECTED
"""
```

---

### Agent A: Controller（控制器）

**功能**：使用 LLM 分析用户输入，决定应该调用哪些专业智能体。

**决策逻辑**：
- 分析用户 prompt 中的关键词和问题类型
- 返回四种决策之一：
  - `"ABC"`: 只涉及 ABC 乐谱技术问题
  - `"EMOTION"`: 只涉及情感分类问题
  - `"BOTH"`: 同时涉及两类问题
  - `"NONE"`: 不涉及任何专业领域

**Prompt 设计**：
```python
def controller_prompt(user_prompt):
    return f"""
You are the Controller Agent.
Your job is to decide which specialized agents should be used.

Rules:
- ABC notation, structure, bars, keys, meter -> "ABC"
- Emotion, mood, valence, arousal, Q1/Q2/Q3/Q4 -> "EMOTION"
- Both topics -> "BOTH"
- Neither -> "NONE"

Return ONLY ONE WORD: ABC, EMOTION, BOTH, or NONE.
"""
```

**关键实现**：
- 使用 `temperature=0` 确保决策一致性
- 包含 fallback 逻辑处理异常响应
- 自动标准化输出（转大写）

---

### Agent B: ABC Expert System（ABC 专家系统）

**功能**：分析 ABC 乐谱并回答关于乐谱结构、调性、节拍等技术问题。

**工作流程**：

1. **ABC 提取**：从用户 prompt 中提取 ABC 乐谱
   ```python
   def extract_abc_from_prompt(user_prompt):
       # 支持多种格式：
       # - ```abc code```
       # - Input: ... Task: ...
       # - Score: ... Task: ...
   ```

2. **ABC Expert 分析**：
   - 分析乐谱的各个组件（调性、节拍、和弦等）
   - 生成结构化分析报告
   - **不直接回答问题**，只提供分析

3. **Evaluator 回答问题**：
   - 基于 ABC Expert 的分析
   - 回答用户的具体问题
   - 支持选项索引提取（0, 1, 2, 3...）

**Prompt 设计**：

**ABC Expert Prompt**:
```python
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
```

**Evaluator Prompt**:
```python
def evaluator_prompt(analysis, task_prompt):
    return f"""
You are the evaluator agent.
You will receive an analysis of an ABC score from the ABC Expert.
Your job is to answer the user's question based ONLY on that analysis.

ABC Expert Analysis:
{analysis}

Task:
{task_prompt}

Important:
- If the question asks for a specific option index, output ONLY that number.
- If the question asks for a general answer, provide a clear answer based on the analysis.
- Base your answer ONLY on the ABC Expert Analysis provided above.
"""
```

**选项索引提取**：
系统使用正则表达式从模型响应中提取选项索引，支持多种格式：
- `**2.` 或 `**2**`
- `2.` 或 `2)` 或 `2 -`
- 简单数字 `2`

---

### Input Validator（输入验证器）

**功能**：在 Controller 之前验证用户输入，确保输入包含有效的 ABC 乐谱和明确的问题。

**工作流程**：

1. **脚本提取 ABC**：首先使用正则表达式尝试从 prompt 中提取 ABC 乐谱

2. **LLM 验证**：
   - 如果未提取到 ABC：调用 LLM 尝试提取，如果确认没有则返回错误
   - 如果已提取到 ABC：调用 LLM 验证 ABC 是否完整，并检查是否包含问题

3. **验证结果**：
   - `VALID_INPUT`: 输入有效，继续处理
   - `NO_ABC_SCORE`: 未检测到 ABC 乐谱，要求用户重新输入
   - `INCOMPLETE_ABC`: ABC 乐谱不完整，提示用户补充
   - `NO_QUESTION_DETECTED`: 未检测到问题，要求用户提问

**Prompt 设计**：

```python
def input_validator_prompt(user_prompt, extracted_abc, has_abc):
    if not has_abc:
        # 尝试提取 ABC
        return f"""You are an input validator...
        [检查是否有 ABC 乐谱]"""
    else:
        # 验证 ABC 完整性和问题存在
        return f"""You are an input validator...
        [验证 ABC 是否完整，检查是否有问题]"""
```

---

### Agent C: Emotion System（情感分类系统）

**功能**：使用基于 arousal-valence 二维模型的多智能体方法对音乐进行情感分类。

**工作流程**：

1. **ABC 提取**：从 prompt 中提取 ABC 乐谱

2. **Arousal 分类（3 个 Analyst）**：
   - 每个 Analyst 独立判断 arousal 是 HIGH 还是 LOW
   - 关注可观察特征：节奏密度、运动连续性、纹理活动、音域活动
   - 使用 `temperature=0.4` 平衡准确性和多样性
   - 输出格式：`AROUSAL: <HIGH/LOW>` + `REASON: <解释>`
   - 使用 majority vote 确定最终 arousal

3. **Valence 分类（3 个 Analyst）**：
   - 每个 Analyst 独立判断 valence 是 HIGH 还是 LOW
   - 关注可观察特征：和声色彩、旋律轮廓、音域亮度、整体亮度/暗度
   - 使用 `temperature=0.4` 平衡准确性和多样性
   - 输出格式：`VALENCE: <HIGH/LOW>` + `REASON: <解释>`
   - 使用 majority vote 确定最终 valence

4. **Combiner 组合决策（1 个 LLM）**：
   - 接收 arousal 和 valence 的分类结果
   - 将两个维度组合成最终的情感类别（Q1-Q4）
   - 使用 `temperature=0.0` 确保一致性
   - 映射规则：
     - High Valence + High Arousal → Q1 (happy) - Label 0
     - Low Valence + High Arousal → Q2 (angry) - Label 1
     - Low Valence + Low Arousal → Q3 (sad) - Label 2
     - High Valence + Low Arousal → Q4 (relaxed) - Label 3

**情感类别定义**：
```
0: Q1 (happy   - high valence, high arousal)
1: Q2 (angry   - low  valence, high arousal)
2: Q3 (sad     - low  valence, low  arousal)
3: Q4 (relaxed - high valence, low  arousal)
```

**Prompt 设计**：

**Arousal Classifier Prompt**:
```python
def build_arousal_classifier_prompt(abc_score):
    return f"""You are an arousal classifier for musical scores written in ABC notation.

Arousal refers to the level of activation or energy in the music:
- HIGH arousal: energetic, intense, driving, highly active
- LOW arousal: calm, peaceful, relaxed, subdued

Guidelines:
- Focus primarily on OBSERVABLE features:
  • rhythmic density (many short note values vs. many long sustained notes)
  • continuity of motion (constant movement vs. frequent pauses)
  • textural activity (thick / busy textures vs. sparse / thin textures)
  • registral activity (frequent leaps and wide range vs. narrow range)

- HIGH arousal is suggested by:
  • many short notes (e.g., 1/8, 1/16) and few long sustained notes
  • continuous motion with little silence
  • frequent leaps, large interval jumps, or rapid figurations
  • dense textures or many notes sounding close together in time

- LOW arousal is suggested by:
  • predominantly long note values and sustained tones
  • slow-moving lines with few changes per bar
  • sparse textures and clear space between events
  • gentle, stepwise motion without much registral excitement

Important constraints:
- Do NOT assume tempo from the meter (e.g., 3/4 is NOT automatically slow).
- Do NOT infer dynamics or performance style (e.g., "soft", "peaceful") unless explicitly indicated.
- Do NOT use major/minor key or mode to decide arousal.

Output format:
Line 1: AROUSAL: <HIGH or LOW>
Line 2: REASON: <1–3 sentences referring to rhythmic density, motion, texture, and range>

Score:
{abc_score}
"""
```

**Valence Classifier Prompt**:
```python
def build_valence_classifier_prompt(abc_score):
    return f"""You are a valence classifier for musical scores written in ABC notation.

Valence refers to the pleasantness or emotional positivity of the music:
- HIGH valence: pleasant, bright, joyful, cheerful
- LOW valence: unpleasant, dark, sad, tense, gloomy

Guidelines:
- Focus on these OBSERVABLE aspects:
  • harmonic color (consonant vs. dissonant, stable vs. tense)
  • melodic contour (upward / soaring vs. downward / sighing or falling)
  • registral brightness (more high-register activity vs. heavy low-register focus)
  • overall sense of brightness or darkness implied by intervals and chord patterns

- HIGH valence is suggested by:
  • predominantly consonant or stable harmonies
  • frequent upward gestures or soaring lines
  • active use of mid–high registers that feel bright or open
  • melodic and harmonic motion that feels flowing or uplifting rather than heavy

- LOW valence is suggested by:
  • frequent dissonance or strong harmonic tension without clear resolution
  • many downward or sinking gestures
  • heavy emphasis on low registers and dark intervallic patterns
  • motion that feels weighed down, unstable, or persistently tense

Important constraints:
- Do NOT treat "minor key = automatically low valence" or "major key = automatically high valence".
  Mode can slightly bias valence, but it is not decisive on its own.
- Do NOT use meter (e.g., 3/4) or assumed tempo to decide valence.
- Do NOT infer emotions that contradict the observable harmonic and melodic features.

Output format:
Line 1: VALENCE: <HIGH or LOW>
Line 2: REASON: <1–3 sentences referring to harmony, contour, and register>

Score:
{abc_score}
"""
```

**Combiner Prompt**:
```python
def build_emotion_combiner_prompt(abc_score, arousal_result, valence_result):
    return f"""You are an emotion classifier that combines arousal and valence dimensions.

You have received two independent classifications:
1. Arousal classification: {arousal_result}
2. Valence classification: {valence_result}

Based on these two dimensions, determine the final emotion category:
- High Valence + High Arousal → Q1 (happy) - Label 0
- Low Valence + High Arousal → Q2 (angry) - Label 1
- Low Valence + Low Arousal → Q3 (sad) - Label 2
- High Valence + Low Arousal → Q4 (relaxed) - Label 3

Your answer (ONLY one number 0/1/2/3):"""
```

**输出格式**：
```
Emotion Classification: Q1 (happy) (Label: 0)

Arousal Classification: HIGH
  Reasoning: The score features dense rhythmic activity with many 1/16 notes...

Valence Classification: HIGH
  Reasoning: The melody has upward gestures and consonant harmonies...

Combined Result: HIGH arousal + HIGH valence → Q1 (happy)
```

---

### Agent D: Aggregator（聚合器）

**功能**：将多个智能体的答案聚合成最终输出。

**实现**：
```python
def agent_D_aggregator(answer_B=None, answer_C=None):
    text = ""
    if answer_B:
        text += f"🎼 **ABC Score Expert Answer:**\n{answer_B}\n\n"
    if answer_C:
        text += f"🎵 **Emotion Expert Answer:**\n{answer_C}\n\n"
    if not text:
        text = "No specialized agent was required. No additional information."
    return text
```

---

### Task Splitter（任务拆分器）

**功能**：当 Controller 决定是 "BOTH" 时，将原始 prompt 拆分为两个独立任务。

**工作流程**：

1. 使用 LLM 分析原始 prompt
2. 提取 ABC 相关部分 → ABC Task
3. 提取情感相关部分 → Emotion Task
4. 确保两个任务都包含必要的 ABC 乐谱信息

**Prompt 设计**：
```python
split_prompt = f"""
You are a task splitter. Given a user prompt that contains both ABC notation questions and emotion classification questions, split it into two separate tasks.

Original prompt:
{user_prompt}

Extract and format:
1. ABC Task: The part related to ABC notation, music structure, keys, meters, bars, chords, etc.
2. Emotion Task: The part related to emotion, mood, valence, arousal, Q1/Q2/Q3/Q4 classification, etc.

Format your response as:
ABC_TASK:
[the ABC-related task here, including the ABC score if present]

EMOTION_TASK:
[the emotion-related task here, including the ABC score if present]

If the original prompt contains an ABC score, include it in BOTH tasks.
"""
```

---

## 数据格式

### 输入数据格式

系统支持两种输入格式：

#### 1. CSV 文件格式（批处理模式）

CSV 文件必须包含 `prompt` 列，格式如下：

**Emotion Recognition 格式**：
```csv
solution,prompt
3,"Input:
X:1
M:3/4
L:1/16
K:Bm
A3G,2DG2< d2[e2A,2] |EAf aB,0<[a0f0] A3[f3F3] |

Task:
Choose the most probable emotional label of the provided score. Label Q1 refers to happy (high valence high arousal), Q2 refers to angry (low valence high arousal), Q3 refers to sad (low valence low arousal) and Q4 refers to relaxed (high valence low arousal).

Options:
0. Q1      1. Q2
2. Q3      3. Q4

Answer:"
```

**Metadata QA 格式**：
```csv
solution,prompt
1,"Input:
X:1
T:Test
K:C
M:4/4
CDEF GABc|

Task:
What is the key of this score?

Options:
0. C   1. D
2. G   3. E

Answer:"
```

#### 2. 交互式输入格式

直接输入包含 ABC 乐谱和问题的文本，例如：
```
Input:
X:1
T:Happy Song
K:C
M:4/4
CDEF GABc|

Task: What emotion does this music express?
Options:
0. Q1      1. Q2
2. Q3      3. Q4
```

### 数据预处理

系统包含 `data/prepare_data.py` 脚本用于预处理原始数据：

1. **Error Detection**: 解析错误列表，构建 prompt
2. **Metadata QA**: 处理选项列表，格式化 prompt
3. **Emotion Recognition**: 添加情感分类选项
4. **Bar Sequencing**: 解析小节选择列表

---

## 安装与配置

### 1. 环境要求

- Python 3.8+
- 虚拟环境（推荐）

### 2. 安装依赖

```bash
# 创建虚拟环境
python -m venv venv

# 激活虚拟环境
# macOS/Linux:
source venv/bin/activate
# Windows:
venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt
```

**依赖列表** (`requirements.txt`):
```
globus-sdk>=3.0.0
openai>=1.0.0
pandas>=1.5.0
scipy>=1.9.0
```

### 3. 认证配置

系统使用 Globus 认证访问 LLM API。首次运行需要认证：

```bash
python inference_auth_token.py authenticate
```

这会：
1. 打开浏览器进行 Globus 登录
2. 要求使用特定域名的账户（`anl.gov`, `alcf.anl.gov`, `uchicago.edu`）
3. 保存 token 到 `~/.globus/app/.../tokens.json`

### 4. API 配置

系统使用 ALCF (Argonne Leadership Computing Facility) 的推理 API：

```python
client = OpenAI(
    api_key=get_access_token(),
    base_url="https://inference-api.alcf.anl.gov/resource_server/sophia/vllm/v1"
)
```

**支持的模型**：
- `google/gemma-3-27b-it` (默认)
- `meta-llama/Meta-Llama-3.1-8B-Instruct`
- `openai/gpt-oss-20b`

修改模型：在代码中更改 `model_name` 变量。

---

## 运行指南

### 方式 1: 交互式模式

```bash
python multi_agent_system.py
```

然后输入问题，例如：
```
Input:
X:1
T:Test
K:C
M:4/4
CDEF GABc|

Task: What emotion does this music express?
Options:
0. Q1      1. Q2
2. Q3      3. Q4
```

输入 `quit` 或 `exit` 退出。

### 方式 2: 批处理模式

```bash
# 处理整个数据集
python multi_agent_system.py data/Emotion_Recognition_cleaned.csv

# 结果会保存为
# data/Emotion_Recognition_cleaned_multi_agent_results.csv
```

### 方式 3: 在代码中调用

```python
from multi_agent_system import run_agent_system

user_prompt = """
Input:
X:1
T:Happy Song
K:C
M:4/4
CDEF GABc|

Task: What emotion does this music express?
Options:
0. Q1      1. Q2
2. Q3      3. Q4
"""

answer = run_agent_system(user_prompt)
print(answer)
```

### 单独运行各个组件

#### 运行情感分类系统（完整版）
```bash
# 测试运行（前10个样本）
python emotion_recognition_agent_2.py --10

# 完整运行
python emotion_recognition_agent_2.py
```

#### 运行元数据 QA 系统
```bash
python metadata_QA_agent.py
```

---

## 代码结构

### 主要文件

```
Agent-for-symbolic-music-understanding/
├── multi_agent_system.py          # 主系统（多智能体协调）
├── emotion_recognition_agent_2.py  # 情感分类系统（独立运行）
├── emotion_recognition_agent.py   # 情感分类系统（旧版本）
├── emotion_recognition_baseline.py # 情感分类基线
├── metadata_QA_agent.py           # 元数据 QA 系统
├── metadata_QA_baseline.py        # 元数据 QA 基线
├── inference_auth_token.py        # Globus 认证模块
├── requirements.txt               # 依赖列表
├── data/
│   ├── prepare_data.py            # 数据预处理脚本
│   ├── Emotion_Recognition_cleaned.csv
│   ├── Metadata_QA_cleaned.csv
│   └── Error_Detection_cleaned.csv
└── venv/                          # 虚拟环境
```

### 核心函数映射

| 功能 | 函数名 | 位置 |
|------|--------|------|
| 主入口 | `run_agent_system()` | `multi_agent_system.py` |
| Controller | `agent_A_controller()` | `multi_agent_system.py` |
| ABC 系统 | `agent_B_abc_system()` | `multi_agent_system.py` |
| 情感系统 | `agent_C_emotion_system()` | `multi_agent_system.py` |
| 任务拆分 | `split_tasks_for_agents()` | `multi_agent_system.py` |
| ABC 提取 | `extract_abc_from_prompt()` | `multi_agent_system.py` |
| 选项提取 | `extract_option_index()` | `multi_agent_system.py` |

---

## 关键实现细节

### 1. ABC 乐谱提取

系统使用多层次的提取策略：

```python
def extract_abc_from_prompt(user_prompt):
    # 1. 尝试提取 ```code``` 格式
    # 2. 尝试 "Input:" 到 "Task:" 之间的内容
    # 3. 尝试 "Score:" 到 "Task:" 之间的内容
    # 4. Fallback: 识别 ABC 标记（X:, K:, M: 等）
    # 5. 最后 fallback: 返回整个 prompt
```

**支持的格式**：
- Markdown code blocks: ` ```abc ... ``` `
- Input format: `Input:\nABC\n\nTask:...`
- Score format: `Score:\nABC\n\nTask:...`
- 直接 ABC 代码

### 2. 选项索引提取

使用正则表达式匹配多种格式：

```python
def extract_option_index(pred_raw, num_options=10):
    # 1. "**2." 或 "**2**"
    # 2. "2." 或 "2)" 或 "2 -"
    # 3. 简单数字 "2"
    # 4. 验证索引 < num_options
```

### 3. 错误处理

- **LLM 调用失败**：返回空字符串或错误信息
- **ABC 提取失败**：提供详细的错误提示
- **选项提取失败**：返回完整响应作为 fallback
- **任务拆分失败**：使用原始 prompt 作为 fallback

### 4. 温度参数设置

| 组件 | Temperature | 原因 |
|------|-------------|------|
| Input Validator | 0.0 | 验证应该准确一致 |
| Controller | 0.0 | 确保决策一致性 |
| ABC Expert | 0.0 | 分析应该客观准确 |
| Evaluator | 0.0 | 答案应该确定 |
| Arousal Analysts | 0.4 | 平衡准确性和多样性，让每个 Analyst 有不同观点 |
| Valence Analysts | 0.4 | 平衡准确性和多样性，让每个 Analyst 有不同观点 |
| Emotion Combiner | 0.0 | 最终决策应该一致 |

### 5. Token 限制

| 组件 | Max Tokens | 原因 |
|------|-----------|------|
| Input Validator | 300 | 可能需要提取或验证 ABC 乐谱 |
| Controller | 默认 | 只需要一个单词 |
| ABC Expert | 默认 | 分析可能较长 |
| Evaluator | 50-200 | 根据是否需要完整答案 |
| Arousal Analysts | 128 | 需要 AROUSAL + REASON |
| Valence Analysts | 128 | 需要 VALENCE + REASON |
| Emotion Combiner | 4 | 只需要一个数字 |

---

## 复现指南

### 步骤 1: 环境设置

```bash
# 克隆或下载项目
cd Agent-for-symbolic-music-understanding

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt
```

### 步骤 2: 认证配置

```bash
# 运行认证脚本
python inference_auth_token.py authenticate

# 验证 token
python inference_auth_token.py get_access_token
```

### 步骤 3: 数据准备（可选）

如果使用自己的数据：

```bash
# 运行数据预处理脚本
python data/prepare_data.py
```

这会生成清理后的 CSV 文件。

### 步骤 4: 测试运行

```bash
# 交互式模式测试
python multi_agent_system.py

# 或批处理模式（小样本）
python emotion_recognition_agent_2.py --5
```

### 步骤 5: 完整运行

```bash
# 运行多智能体系统
python multi_agent_system.py data/Emotion_Recognition_cleaned.csv

# 或运行独立的情感分类系统
python emotion_recognition_agent_2.py
```

### 步骤 6: 结果分析

结果会保存在 CSV 文件中，包含：
- 原始数据
- 预测结果
- 原始响应（用于调试）

---

## 使用示例

### 示例 1: 情感分类

**输入**：
```
Input:
X:1
T:Happy Song
K:C
M:4/4
L:1/8
CDEF GABc|CDEF GABc|

Task: What emotion does this music express?
Options:
0. Q1      1. Q2
2. Q3      3. Q4
```

**处理流程**：
1. Controller 识别为 "EMOTION"
2. Agent C 提取 ABC 乐谱
3. 3 个 Analyst 独立分类
4. 1 个 Judge 综合决策

**输出**：
```
🎵 **Emotion Expert Answer:**
Emotion Classification: Q1 (happy) (Label: 0)

Analyst Predictions:
  Analyst 1: Q1 (happy) (Label: 0) - The score is in C major with a bright, upbeat melody...
  Analyst 2: Q1 (happy) (Label: 0) - Fast tempo and major key suggest high energy and positive emotion...
  Analyst 3: Q1 (happy) (Label: 0) - The ascending melody and major tonality indicate happiness...
```

### 示例 2: ABC 技术问题

**输入**：
```
Input:
X:1
T:Test
K:D
M:3/4
L:1/4
DEF GAB|

Task: What is the key of this score?
Options:
0. C   1. D
2. G   3. E
```

**处理流程**：
1. Controller 识别为 "ABC"
2. Agent B 提取 ABC 乐谱
3. ABC Expert 分析乐谱（识别 K:D）
4. Evaluator 基于分析回答问题

**输出**：
```
🎼 **ABC Score Expert Answer:**
Answer: 1

Based on the ABC analysis: The score is in the key of D major, as indicated by the K:D header in the ABC notation.
```

### 示例 3: 混合问题（BOTH）

**输入**：
```
Input:
X:1
T:Complex Analysis
K:Am
M:4/4
CDEF GABc|

Task: 
1. What is the key of this score?
2. What emotion does this music express?

Options for Q1:
0. C   1. Am   2. G   3. E

Options for Q2:
0. Q1   1. Q2   2. Q3   3. Q4
```

**处理流程**：
1. Controller 识别为 "BOTH"
2. Task Splitter 拆分任务：
   - ABC Task: 关于调性的问题
   - Emotion Task: 关于情感的问题
3. Agent B 处理 ABC Task
4. Agent C 处理 Emotion Task
5. Agent D 聚合答案

**输出**：
```
🎼 **ABC Score Expert Answer:**
Answer: 1

The key is A minor (Am), as indicated by the K:Am header.

🎵 **Emotion Expert Answer:**
Emotion Classification: Q3 (sad) (Label: 2)

Analyst Predictions:
  Analyst 1: Q3 (sad) (Label: 2) - Minor key suggests low valence...
  Analyst 2: Q3 (sad) (Label: 2) - The A minor tonality creates a melancholic mood...
  Analyst 3: Q2 (angry) (Label: 1) - While minor, the tempo might suggest higher arousal...
```

---

## 性能优化建议

### 1. 减少 LLM 调用

- 对于简单问题，可以考虑基于规则的 Controller
- 缓存 ABC Expert 的分析结果（如果相同乐谱被多次询问）

### 2. 并行处理

- Emotion Analysts 可以并行调用（需要异步实现）
- 批处理模式可以并行处理多个样本

### 3. Token 优化

- 根据实际需要调整 `max_tokens`
- 对于只需要选项索引的情况，使用较小的 `max_tokens`

### 4. 错误重试

- 实现 LLM 调用的重试机制
- 对于临时性错误，自动重试

---

## 故障排除

### 问题 1: 认证失败

**症状**：`Error: Access token does not exist`

**解决**：
```bash
python inference_auth_token.py authenticate --force
```

### 问题 2: ABC 提取失败

**症状**：`Error: Could not extract ABC score`

**解决**：
- 检查 prompt 格式是否包含 "Input:" 或 "Score:"
- 确保 ABC 代码格式正确
- 查看 `extract_abc_from_prompt()` 函数的 fallback 逻辑

### 问题 3: Controller 返回 "NONE"

**症状**：系统无法识别问题类型

**解决**：
- 在 prompt 中明确包含关键词（如 "emotion", "key", "meter"）
- 检查 `controller_prompt` 的设计
- 查看 Controller 的 fallback 逻辑

### 问题 4: 选项提取失败

**症状**：返回完整文本而不是选项索引

**解决**：
- 检查 `extract_option_index()` 的正则表达式
- 查看模型响应格式
- 考虑调整 prompt 要求更明确的格式

---

## 扩展建议

### 1. 添加新的 Agent

要添加新的专业智能体（如 Agent E）：

1. 实现新的 agent 函数：
```python
def agent_E_new_system(user_prompt):
    # 实现逻辑
    return answer
```

2. 更新 Controller prompt，添加新的识别规则

3. 更新 `run_agent_system()`，添加新的路由逻辑

4. 更新 Aggregator，添加新的答案格式

### 2. 支持更多数据格式

扩展 `extract_abc_from_prompt()` 支持：
- MusicXML 格式
- MIDI 文件
- 其他符号音乐格式

### 3. 添加 Few-shot Learning

为 Emotion Analysts 添加 few-shot examples：
```python
fewshot_examples = build_fewshot(df, k=6)
analyst_prompt += f"\n\nExamples:\n{fewshot_examples}"
```

### 4. 实现异步处理

使用 `asyncio` 并行调用多个 Analysts：
```python
import asyncio

async def call_analyst_async(prompt):
    # 异步调用 LLM
    pass

analyst_tasks = [call_analyst_async(p) for p in prompts]
results = await asyncio.gather(*analyst_tasks)
```

---

## 参考文献与资源

### ABC 乐谱格式
- [ABC Notation Standard](http://abcnotation.com/)
- ABC 是一种文本格式的音乐记谱法

### 情感分类理论
- Valence-Arousal 二维情感模型
- Q1-Q4 四象限分类法

### 多智能体系统
- Multi-Agent Systems 架构模式
- LLM-based Agent Coordination

---

## 许可证与致谢

本项目使用特定的 LLM API（ALCF），需要相应的访问权限。

**致谢**：
- ALCF (Argonne Leadership Computing Facility) 提供推理 API
- Globus 提供认证服务

---

## 更新日志

### v1.0 (当前版本)
- ✅ 实现多智能体系统架构
- ✅ Agent A: LLM-based Controller
- ✅ Agent B: ABC Expert System
- ✅ Agent C: Emotion Classification System (3 Analysts + 1 Judge)
- ✅ Agent D: Answer Aggregator
- ✅ Task Splitter for BOTH scenarios
- ✅ 支持交互式和批处理模式
- ✅ 完整的错误处理

---

## 联系方式

如有问题或建议，请通过以下方式联系：
- 提交 Issue
- 发送 Pull Request

---

**文档版本**: 1.0  
**最后更新**: 2024  
**维护者**: [Your Name]

