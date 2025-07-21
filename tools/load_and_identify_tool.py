"""
Tool: load_and_identify_tool

Combines dataset loading and ML task/target identification.
Uses PyCaret's built-in datasets and Groq LLM.

Steps:
1. Load dataset from name using load_dataset_tool
2. Call task_identifier_tool to infer task + target
3. Return combined message + await user confirmation

Dependencies:
    - load_dataset_tool
    - task_identifier_tool

Updates:
    ml_state["data"], ml_state["task"], ml_state["target"]


from tools.load_dataset_tool import load_dataset, AVAILABLE_DATASETS
from llm.task_identifier import task_identifier_tool
from state.ml_state import ml_state
from langchain_core.tools import tool

#@tool(return_direct=True)
def load_and_identify_tool(dataset_name: str, user_query: str) -> str:
    
    dataset_name = dataset_name.lower().replace(".csv", "")

    if dataset_name not in AVAILABLE_DATASETS:
        return f"\u274c Dataset '{dataset_name}' not found.\n\nAvailable datasets: {', '.join(AVAILABLE_DATASETS)}"

    # Step 1: Load Dataset
    load_result = load_dataset(dataset_name)
    if ml_state["data"] is None:
        return f"\u274c Failed to load dataset '{dataset_name}'."

    # Step 2: Pass to LLM task identifier
    try:
        llm_response = task_identifier_tool(ml_state["data"], user_query)
        return f"{load_result}\n\n{llm_response}"
    except Exception as e:
        return f"\u274c Failed to analyze dataset using LLM: {str(e)}"
"""