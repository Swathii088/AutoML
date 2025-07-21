import logging
import json
import pandas as pd # Import pandas for DataFrame operations
from typing import Dict, Any, Optional
from pycaret.classification import tune_model as classification_tune, pull as classification_pull
from pycaret.regression import tune_model as regression_tune, pull as regression_pull
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from state.graph_state import MLState 
from dotenv import load_dotenv
load_dotenv()

# Configure logging and LLM
logger = logging.getLogger(__name__)
llm = ChatGroq(
    temperature=0.0,
    model_name="llama3-70b-8192",
    model_kwargs={"response_format": {"type": "json_object"}},
)

# --- Configuration ---

VALID_OPTIMIZE_METRICS = {
    "classification": ["Accuracy", "AUC", "Recall", "Precision", "F1", "Kappa", "MCC"],
    "regression": ["MAE", "MSE", "RMSE", "R2", "RMSLE", "MAPE"]
}

DEFAULT_TUNE_CONFIG = {
    "n_iter": 10,
    "search_library": "scikit-learn",
    "search_algorithm": "random",
    "choose_better": True,
    "verbose": False # Ensure PyCaret doesn't print directly to console
}

def _get_llm_tune_parameters(user_query: str, task: str) -> Dict[str, Any] | None:
    """
    Uses an LLM to parse the user query for tune_model parameters.
    Expected JSON output: {"optimize": "metric", "n_iter": int, ...}
    """
    optimize_options = ", ".join(VALID_OPTIMIZE_METRICS[task])
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", f"""
You are a machine learning assistant. Extract parameters from the user's query for PyCaret's `tune_model()` function.

Parameters to extract:
- optimize (str): The metric to optimize. MUST be one of: {optimize_options}.
- n_iter (int): Number of iterations for the search. Default is 10.
- search_library (str): The library to use for tuning ('scikit-learn', 'optuna', 'tune-sklearn'). Default is 'scikit-learn'.
- search_algorithm (str): The search algorithm ('random', 'grid', 'tpe', 'bayesian', 'hyperopt'). Default is 'random'.

Return ONLY a valid JSON object. Do not add explanations or markdown.
If a parameter is not specified by the user, do not include it in the JSON.
"""),
        ("user", "User Query: {user_query}")
    ])
    chain = prompt_template | llm
    try:
        response = chain.invoke({
            "optimize_options": optimize_options,
            "user_query": user_query
        })
        return json.loads(response.content.strip())
    except json.JSONDecodeError as e:
        logger.error(f"LLM did not return a valid JSON object for tune_model parameters: {e}")
        return None
    except Exception as e:
        logger.error(f"LLM call for tune_model parameters failed: {e}")
        return None

@tool("tune_model_tool", return_direct=True) # Added return_direct=True
def tune_model_tool(state: MLState, user_query: Optional[str] = None) -> Dict[str, Any]:
    """
    Performs hyperparameter tuning on the current model in the state.

    This tool intelligently selects the best available model from the state
    (prioritizing `created_model` or `best_model` if it's a single model) and 
    optimizes its hyperparameters. It can parse natural language for tuning 
    parameters like 'n_iter' or the optimization metric 
    (e.g., 'tune the model for 50 iterations optimizing for AUC').

    This tool is currently applicable only for classification and regression tasks.

    Args:
        state (MLState): The current state of the ML pipeline, which must include
                         a model to be tuned (either `created_model` or `best_model`).
        user_query (Optional[str]): The user's command with tuning preferences.

    Returns:
        Dict[str, Any]: A dictionary containing the `tuned_model`, an updated main `model`
                        reference, and the `last_output` with tuning results to update the state.
    """
    logger.info("--- Executing Tune Model Tool ---")

    # Determine which model to tune: prioritize created_model, then best_model (if single)
    model_to_tune = None
    if state.created_model:
        model_to_tune = state.created_model
    elif state.best_model and not isinstance(state.best_model, list): # Ensure best_model is a single model, not a list from n_select > 1
        model_to_tune = state.best_model
    elif state.best_model and isinstance(state.best_model, list) and len(state.best_model) > 0:
        model_to_tune = state.best_model[0] # Take the first model from a list if multiple were selected

    if not state.setup_done or model_to_tune is None:
        return {"last_output": "❌ Please run setup and create/compare models before tuning. No suitable model found in state."}
    
    task = state.task
    if task not in ("classification", "regression"):
        return {"last_output": f"❌ `tune_model` tool is not applicable for the '{task}' task. It is only for classification and regression."}

    # Set default optimization metric based on task
    config = DEFAULT_TUNE_CONFIG.copy()
    config["optimize"] = "F1" if task == "classification" else "R2"
    
    # Ensure verbose is False by default for automated execution
    config["verbose"] = False

    if user_query:
        logger.info(f"Parsing user query for tune_model parameters: '{user_query}' for task '{task}'")
        llm_updates = _get_llm_tune_parameters(user_query, task)
        if llm_updates:
            # Apply updates, ensuring 'optimize' metric is valid if provided
            if 'optimize' in llm_updates and llm_updates['optimize'] not in VALID_OPTIMIZE_METRICS[task]:
                logger.warning(f"Invalid optimize metric '{llm_updates['optimize']}' provided. Ignoring optimize parameter.")
                llm_updates.pop('optimize', None) # Remove invalid optimize key
            
            # Sanitize to only include allowed keys
            sanitized_updates = {k: v for k, v in llm_updates.items() if k in DEFAULT_TUNE_CONFIG or k == 'optimize'}
            config.update(sanitized_updates)
            logger.info(f"Applied LLM-driven config updates: {sanitized_updates}")

    logger.info(f"Final tune_model config: {config}")

    tune_map = {"classification": classification_tune, "regression": regression_tune}
    pull_map = {"classification": classification_pull, "regression": regression_pull}

    try:
        tune_function = tune_map[task]
        tuned_model = tune_function(model_to_tune, **config) # verbose is already False in config
        
        # Pull the performance metrics DataFrame
        tuning_results = pull_map[task]()
        
        logger.info("Successfully tuned model.")
        
        return {
            "tuned_model": tuned_model,
            "model": tuned_model, # Also update the main 'model' reference
            "last_output": f"✅ Model tuning complete. Tuned model is now the active model.\n\n**Tuning Results:**\n```\n{tuning_results.to_string()}\n```"
        }
    except Exception as e:
        logger.error(f"PyCaret tune_model failed: {e}", exc_info=True)
        return {"last_output": f"❌ Failed to tune model: {e}. Please check the model or tuning parameters."}