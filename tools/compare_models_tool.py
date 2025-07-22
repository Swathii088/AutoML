import logging
import json
from typing import Dict, Any, Optional
from pycaret.classification import compare_models as classification_compare, pull as classification_pull
from pycaret.regression import compare_models as regression_compare, pull as regression_pull
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

VALID_SORT_METRICS = {
    "classification": ["Accuracy", "AUC", "Recall", "Precision", "F1", "Kappa", "MCC"],
    "regression": ["MAE", "MSE", "RMSE", "R2", "RMSLE", "MAPE"]
}

def _get_llm_compare_parameters(user_query: str, task: str) -> Dict[str, Any] | None:
    """Uses an LLM to parse the user query into a dictionary of compare_models parameters."""
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", """
You are a machine learning assistant. Extract values from the user's query and return a JSON object containing parameters for PyCaret's `compare_models()` function.

Allowed parameters:
- n_select (int): The number of top models to return.
- sort (str): The metric to sort by. Must be one of: {sort_options}
- include (list[str]): A list of model IDs to train.
- exclude (list[str]): A list of model IDs to exclude from training.

Return ONLY a valid JSON object. Do not add explanations.
"""),
        ("user", "User Query: {user_query}")
    ])
    chain = prompt_template | llm
    try:
        response = chain.invoke({
            "sort_options": VALID_SORT_METRICS[task],
            "user_query": user_query
        })
        return json.loads(response.content.strip())
    except Exception as e:
        logger.error(f"LLM call for compare_models parameters failed: {e}")
        return None

@tool("compare_models_tool")
def compare_models_tool(state: MLState, user_query: Optional[str] = None) -> Dict[str, Any]:
    """
    Compares multiple ML models and updates the state with the best models and leaderboard.
    
    Args:
        state (MLState): The current state of the ML pipeline.
        user_query (str): Natural language instructions for model comparison.
    
    Returns:
        Dict[str, Any]: A dictionary of updates to be merged into the state.
    """
    logger.info("--- Executing Compare Models Tool ---")

    if not state.setup_done:
        return {"last_output": "❌ Please run the setup tool before comparing models."}

    task = state.task
    if task not in ("classification", "regression"):
        return {"last_output": f"❌ compare_models tool is not applicable for the '{task}' task."}
    
    default_config = {
        "n_select": 1,
        "sort": "F1" if task == "classification" else "R2",
        "fold": 10,
        "verbose" : False
    }
    config = default_config.copy()

    if user_query:
        logger.info(f"Parsing user query for compare_models parameters: '{user_query}'")
        llm_updates = _get_llm_compare_parameters(user_query, task)
        if llm_updates:
            # Sanitize to only include allowed keys
            sanitized_updates = {k: v for k, v in llm_updates.items() if k in config}
            config.update(sanitized_updates)
            logger.info(f"Applied LLM-driven config updates: {sanitized_updates}")

    logger.info(f"Final compare_models config: {config}")

    compare_map = {
        "classification": classification_compare,
        "regression": regression_compare
    }
    pull_map = {
        "classification": classification_pull,
        "regression": regression_pull
    }
    try:
        compare_function = compare_map[task]
        best_models = compare_function(**config)
        
        leaderboard = pull_map[task]()
        
        logger.info("Model comparison successful. Leaderboard retrieved.")
        
        # --- KEY FIX ---
        # Use the 'n_select' value to show only the top N models requested by the user.
        top_n = config.get("n_select", 1)
        leaderboard_preview = leaderboard.head(top_n).to_string()
        
        return {
            "best_model": best_models,
            "model": best_models, # Also update the active model reference
            "leaderboard": leaderboard,
            "last_output": f"✅ Model comparison complete. The best {top_n} model(s) have been selected.\n\n**Top {top_n} Models:**\n```\n{leaderboard_preview}\n```"
        }
    except Exception as e:
        logger.error(f"PyCaret compare_models failed: {e}", exc_info=True)
        return {"last_output": f"❌ compare_models failed: {e}"}
