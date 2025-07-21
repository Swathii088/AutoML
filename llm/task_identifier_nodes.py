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
    """Calls the LLM in JSON mode to get a suggestion for the ML task and target."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert ML assistant. Your goal is to determine the correct machine learning task and target column based on a user's query and a dataset summary. You must respond with a valid JSON object."),
        ("user", """
For supervised tasks ('classification' or 'regression'), you MUST identify the most likely target column from the dataset's columns. For unsupervised tasks, the target MUST be null.

Supported tasks are:
- Supervised: 'classification', 'regression'
- Unsupervised: 'clustering', 'anomaly_detection', 'Time Series'

---
## User Query
{user_query}

## Dataset Summary
- Shape: {shape}
- Columns: {columns}
- Sample Head:
{sample_head}
---

JSON Output:
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
    """Node that calls the LLM to identify the ML task and target column."""
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
    """Node that formats the confirmation message for the user."""
    logger.info("---NODE: REQUEST CONFIRMATION---")
    task = state.task
    target = state.target_column
    
    if task in ['classification', 'regression']:
        message = (
            f"🤖 Based on your request, I identified a **{task}** task "
            f"with **`{target}`** as the target column.\n\n"
            "Is this correct? (Reply 'yes', 'no', or the correct column name)."
        )
    else:
        message = (
            f"🤖 Based on your request, I identified an unsupervised **{task}** task.\n\n"
            "Shall we proceed? (Reply 'yes' or 'no')."
        )
    return {"last_output": message}

def handle_validation_node(state: MLState) -> dict:
    """Node that processes the user's validation response."""
    logger.info("---NODE: HANDLE VALIDATION---")
    user_response = state.user_validation_response.strip().lower()
    task = state.task
    df = state.data
    
    if user_response == 'yes':
        return {"last_output": f"✅ Great! Task confirmed: **{task}**. You can now proceed to setup."}

    if user_response == 'no':
        col_list = "\n".join([f"- `{col}`" for col in df.columns])
        return {"last_output": f"Apologies. Please specify the task and, if needed, the correct target column from this list:\n{col_list}"}
    
    lower_cols = {col.lower(): col for col in df.columns}
    if user_response in lower_cols:
        new_target = lower_cols[user_response]
        return {
            "target_column": new_target,
            "last_output": f"✅ Understood. Target column updated to **`{new_target}`**."
        }
    
    return {"last_output": "I didn't understand. Please reply with 'yes', 'no', or a valid column name."}