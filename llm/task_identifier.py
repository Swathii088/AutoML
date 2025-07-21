import os
import json
import re
import pandas as pd
from dotenv import load_dotenv
from groq import Groq
from langchain_core.tools import tool
from state.ml_state import ml_state

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def format_prompt(df: pd.DataFrame, user_query: str) -> str:
    column_types = {col: str(dtype) for col, dtype in df.dtypes.items()}
    missing_values = {col: int(df[col].isnull().sum()) for col in df.columns}
    sample_data = df.head(5).to_string(index=False)
    
    return f"""
You are an expert ML assistant.

Analyze the dataset and user request below. Identify:
1. The most appropriate ML task from:
   - classification, regression, clustering, time series, anomaly detection
2. The most likely target column (if applicable)

Respond only in JSON like:
{{
  "task": "<task_type>",
  "target": "<column_name_or_null_if_unsupervised>"
}}

---
User Query:
{user_query}

Dataset Summary:
- Shape: {df.shape}
- Columns: {list(df.columns)}
- Data types: {column_types}
- Missing values: {missing_values}

Sample Data:
{sample_data}

Statistics:
{df.describe(include='all').to_string()}
"""

def run_llm(prompt: str, context: str = "") -> str:
    try:
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": context}
            ],
            model="deepseek-r1-distill-llama-70b",
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"LLM Error: {str(e)}"

def extract_json(text: str) -> str:
    match = re.search(r"\{[\s\S]*?\}", text)
    return match.group(0) if match else text

def task_identifier_tool(df: pd.DataFrame, user_query: str) -> str:
    prompt = format_prompt(df, user_query)
    raw = run_llm(prompt)

    # Extract JSON part
    try:
        result = json.loads(extract_json(raw))
        ml_state.update({
            ml_state["task"]: result["task"],
            ml_state["target"]: result.get("target"),
            "task_confirmed": not result.get("target"),
            "last_identified": result
        })

        # ✅ Return a clean final answer
        if result.get("target"):
            return (
                f"🤖 Suggested task: **{result['task']}**\n"
                f"🎯 Target column: **`{result['target']}`**\n"
                "✅ Shall we proceed? (yes / no / column name)"
            )
        else:
            return (
                f"🤖 Suggested task: **{result['task']}** (unsupervised — no target column needed)\n"
                "✅ You may proceed to setup or modeling."
            )

    except json.JSONDecodeError:
        return f"⚠️ Couldn't parse LLM response. Raw output:\n\n{raw}"

def handle_user_response_to_task_selection(df: pd.DataFrame, user_input: str) -> str:
    response = user_input.strip()

    if response.lower() == "yes":
        ml_state["task_confirmed"] = True
        return f"✅ Task: `{ml_state['task']}`, Target: `{ml_state['target']}`. Run `setup()` to continue."

    if response.lower() == "no":
        cols = "\n".join([f"{i+1}. {col}" for i, col in enumerate(df.columns)])
        return f"❌ Please select the correct target column:\n\n{cols}\n\nReply with the column name or number."

    if response.isdigit():
        idx = int(response) - 1
        if 0 <= idx < len(df.columns):
            ml_state.update({"target": df.columns[idx], "task_confirmed": True})
            return f"✅ Target column updated to `{df.columns[idx]}`. Proceed with setup."
        return f"⚠️ Invalid number. Enter a value between 1 and {len(df.columns)}."

    if response in df.columns:
        ml_state.update({"target": response, "task_confirmed": True})
        return f"✅ Target column set to `{response}`. Proceed with setup."

    return "⚠️ Invalid input. Reply with `yes`, `no`, column name, or column number."

@tool
def identify_task_and_target(user_query: str = None, user_validation_input: str = None) -> str:
    """
    Identify ML task and target column based on dataset and query,
    then handle user validation input for confirmation or corrections.

    Args:
        user_query (str, optional): Natural language task request.
        user_validation_input (str, optional): User response to validate identified task/target.

    Returns:
        str: Suggested ML task and target column (if applicable) or validation handling response.
    """
    df = ml_state.get("data")
    if df is None:
        return "⚠️ No dataset loaded. Use `load_dataset()` first."

    # Step 1: If user_validation_input is not provided, identify task and target
    if user_validation_input is None:
        return task_identifier_tool(df, user_query)

    # Step 2: Otherwise, handle the user validation response
    return handle_user_response_to_task_selection(df, user_validation_input)
    
