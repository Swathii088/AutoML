import os
import pandas as pd
import json
import logging
from typing import Dict, Any, Optional
from pycaret.classification import setup as classification_setup
from pycaret.regression import setup as regression_setup
from pycaret.clustering import setup as clustering_setup
from pycaret.anomaly import setup as anomaly_setup
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from state.graph_state import MLState
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- LLM and Prompting Setup ---
llm = ChatGroq(
    temperature=0.0,
    model_name="llama3-70b-8192",
    model_kwargs={"response_format": {"type": "json_object"}},
)

# REFINEMENT 1: Separate configs for supervised and unsupervised tasks
DEFAULT_SUPERVISED_CONFIG = {
    "profile": False, "fold": 10, "feature_selection": False,
    "remove_outliers": False, "remove_multicollinearity": False, "train_size": 0.7,
    "preprocess": True, "numeric_imputation": "mean", "categorical_imputation": "mode"
}
DEFAULT_UNSUPERVISED_CONFIG = {
    "profile": False, "preprocess": True, "numeric_imputation": "mean",
    "categorical_imputation": "mode", "normalize": False
}

def _get_llm_setup_parameters(user_query: str, available_params: list) -> Dict[str, Any] | None:
    """Uses an LLM to parse the user query into a dictionary of setup parameters."""
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", """You are an expert at extracting parameters from text. Your task is to analyze the user's query and return a JSON object containing only the PyCaret `setup` parameters that the user explicitly mentioned.

- You must only return a valid JSON object.
- Do not include parameters that the user did not mention.
- For boolean parameters, if the user mentions it (e.g., "with outliers" or "remove outliers"), set the value to `true`.

Available parameters: {parameters}
"""),
        ("user", "User Query: {user_query}\n\nJSON Output:")
    ])
    chain = prompt_template | llm
    try:
        response = chain.invoke({
            "parameters": available_params,
            "user_query": user_query
        })
        return json.loads(response.content.strip())
    except Exception as e:
        logger.error(f"LLM call for setup parameters failed: {e}")
        return None

@tool("setup_tool", return_direct=True)
def setup_tool(state: MLState, user_query: Optional[str] = None) -> Dict[str, Any]:
    """
    Initializes the PyCaret environment by running the setup() function.
    This tool performs data preprocessing and prepares the data for modeling.
    It takes preprocessing instructions from a user query and updates the state.
    
    Args:
        state (MLState): The current state of the ML pipeline.
        user_query (str): Natural language instructions for preprocessing (optional).

    Returns:
        Dict[str, Any]: A dictionary of updates to be merged into the state.
    """
    logger.info("--- Executing Setup Tool ---")
    
    data = state.data
    target = state.target_column
    task = state.task
    
    if data is None or task is None:
        return {"last_output": "❌ Cannot run setup. Dataset and task must be selected."}

    is_supervised = task in ["classification", "regression"]
    
    # REFINEMENT 2: Select the correct config based on the task
    config = (DEFAULT_SUPERVISED_CONFIG if is_supervised else DEFAULT_UNSUPERVISED_CONFIG).copy()

    if user_query:
        logger.info(f"Parsing user query for setup parameters: '{user_query}'")
        llm_updates = _get_llm_setup_parameters(user_query, list(config.keys()))
        if llm_updates:
            config.update(llm_updates)
            logger.info(f"Applied LLM-driven config updates: {llm_updates}")

    logger.info(f"Final setup config for '{task}' task: {config}")

    setup_map = {
        "classification": classification_setup,
        "regression": regression_setup,
        "clustering": clustering_setup,
        "anomaly": anomaly_setup,
    }
    
    if task not in setup_map:
        return {"last_output": f"❌ Unsupported task type for setup: {task}."}

    if is_supervised and not target:
        return {"last_output": f"❌ Target column must be set for a {task} task."}
    
    # REFINEMENT 3: Simplified classification pre-run validation
    if task == "classification":
        if data[target].nunique() < 2:
            return {"last_output": "❌ Target column has less than 2 unique classes for classification."}

    try:
        # REFINEMENT 4: Set verbose=False for cleaner programmatic output
        setup_kwargs = {"data": data, "session_id": 123, "verbose": True, **config}
        if is_supervised:
            setup_kwargs["target"] = target

        setup_function = setup_map[task]
        setup_pipeline = setup_function(**setup_kwargs)


        logger.info(f"PyCaret setup successful for {task} task.")
        return {
            "setup_pipeline": setup_pipeline,
            "setup_config": config,
            "setup_done": True,
            "last_output": f"✅ Setup complete for {task} task. Target: '{target if is_supervised else 'N/A'}'\n\n""Setup Configuration:\n{setup_pipeline.to}."
        }
    except Exception as e:
        logger.error(f"PyCaret setup failed: {e}", exc_info=True)
        return {"last_output": f"❌ PyCaret setup failed: {e}"}