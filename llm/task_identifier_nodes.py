import os
import json
import pandas as pd
from state.graph_state import MLState
from typing import Optional, Dict, Any
import logging
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()
logger = logging.getLogger(__name__)

llm = ChatGroq(
    temperature=0.0,
    model_name="llama3-70b-8192",
    model_kwargs={"response_format": {"type": "json_object"}},
)

def _get_llm_task_suggestion(dataframe: pd.DataFrame, user_query: str) -> Optional[Dict[str, Any]]:
    # ... (this function remains the same)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert ML assistant. Your goal is to determine the correct machine learning task and target column based on a user's query and a dataset summary. You must respond with a valid JSON object."),
        ("user", """
For supervised tasks ('classification' or 'regression'), you MUST identify the most likely target column from the dataset's columns. For unsupervised tasks, the target MUST be null.

Supported tasks are:
- Supervised: 'classification', 'regression'
- Unsupervised: 'clustering', 'anomaly_detection'

---
## User Query
{user_query}

## Dataset Summary
- Shape: {shape}
- Columns: {columns}
- Sample Head:
{sample_head}
---

JSON Output (e.g., {{"task": "regression", "target": "medv"}}):
""")
    ])
    chain = prompt | llm
    try:
        response = chain.invoke({
            "user_query": user_query,
            "shape": dataframe.shape,
            "columns": list(dataframe.columns),
            "sample_head": dataframe.head(3).to_string(),
        })
        return json.loads(response.content)
    except Exception as e:
        logger.error(f"LLM call or JSON parsing failed: {e}")
        return None


def identify_task_node(state: MLState) -> dict:
    # ... (this function remains the same)
    logger.info("---NODE: IDENTIFY TASK---")
    df = state.data
    query = state.input_message
    suggestion = _get_llm_task_suggestion(df, query)
    if not suggestion or "task" not in suggestion:
        return {"last_output": "Sorry, I could not determine the ML task."}
    logger.info(f"LLM suggested task: {suggestion.get('task')}, target: {suggestion.get('target')}")
    return {
        "task": suggestion.get("task"),
        "target_column": suggestion.get("target"),
    }

def request_confirmation_node(state: MLState) -> dict:
    # ... (this function remains the same)
    logger.info("---NODE: REQUEST CONFIRMATION---")
    task = state.task
    target = state.target_column
    if task in ['classification', 'regression']:
        message = (
            f"🤖 Based on your request, I identified a **{task}** task "
            f"with **`{target}`** as the target column.\n\n"
            "Is this correct? (Reply 'yes', 'no', or provide the correction)."
        )
    else:
        message = (
            f"🤖 Based on your request, I identified an unsupervised **{task}** task.\n\n"
            "Shall we proceed? (Reply 'yes' or 'no')."
        )
    return {"last_output": message}

# --- UPDATED AND IMPROVED VALIDATION NODE ---
def handle_validation_node(state: MLState) -> dict:
    """
    Processes the user's validation response. If the user provides a correction,
    it uses an LLM to parse the new task and/or target column.
    """
    logger.info("---NODE: HANDLE VALIDATION---")
    user_response = state.user_validation_response.strip().lower()
    df = state.data
    
    if user_response == 'yes':
        return {"last_output": f"✅ Great! Task confirmed: **{state.task}**. You can now proceed."}

    if user_response == 'no':
        col_list = "\n".join([f"- `{col}`" for col in df.columns])
        return {"last_output": f"Apologies. Please specify the correct task and/or target column from this list:\n{col_list}"}
    
    # If the user provides a correction, use an LLM to parse it
    logger.info(f"User provided correction: '{user_response}'. Parsing with LLM.")
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert at parsing user corrections for an ML task. Extract the `task` and `target_column` from the user's message. The task must be one of ['classification', 'regression', 'clustering', 'anomaly_detection']. The target_column must be one of the available columns. If a value isn't mentioned, set it to null."),
        ("user", "Available columns: {columns}\n\nUser Correction: {correction}\n\nJSON Output:")
    ])
    chain = prompt | llm
    try:
        response = chain.invoke({
            "columns": list(df.columns),
            "correction": user_response
        })
        parsed_correction = json.loads(response.content)
        
        updates = {}
        # Update only if the LLM provides a new value
        if parsed_correction.get("task"):
            updates["task"] = parsed_correction["task"]
        if parsed_correction.get("target_column"):
            updates["target_column"] = parsed_correction["target_column"]
            
        if not updates:
             return {"last_output": "I didn't understand the correction. Please be more specific, for example: 'it is classification with the species column'."}

        final_task = updates.get("task", state.task)
        final_target = updates.get("target_column", state.target_column)

        return {
            **updates,
            "last_output": f"✅ Understood. Task updated to **{final_task}** and target to **{final_target}**. You can now proceed."
        }
    except Exception as e:
        logger.error(f"Failed to parse user correction: {e}")
        return {"last_output": "Sorry, I had trouble understanding your correction. Please try again."}

