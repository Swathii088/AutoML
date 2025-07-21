import logging
import json
import pandas as pd # Import pandas for DataFrame operations
from typing import Dict, Any, Optional
from pycaret.classification import ensemble_model as classification_ensemble, pull as classification_pull
from pycaret.regression import ensemble_model as regression_ensemble, pull as regression_pull
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from state.graph_state import MLState # Adjust import path as needed
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

DEFAULT_ENSEMBLE_CONFIG = {
    "method": "Bagging",
    "n_estimators": 10,
    "choose_better": False,
    "verbose": False 
}

def _get_llm_ensemble_parameters(user_query: str) -> Dict[str, Any] | None:
    """
    Uses an LLM to parse the user query for ensemble_model parameters.
    Expected JSON output: {"method": "Bagging" | "Boosting", "n_estimators": int, "choose_better": bool}
    """
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", """
You are a machine learning assistant. Extract parameters from the user's query for PyCaret's `ensemble_model()` function.

Parameters to extract:
- method (str): The ensembling method. MUST be either 'Bagging' or 'Boosting'. Default is 'Bagging'.
- n_estimators (int): The number of base estimators (models) to use in the ensemble. Default is 10.
- choose_better (bool): Whether to return the better of the base model and the ensembled model. Default is false.

Return ONLY a valid JSON object. Do not add explanations or markdown.
If a parameter is not specified by the user, do not include it in the JSON.
"""),
        ("user", "User Query: {user_query}")
    ])
    chain = prompt_template | llm
    try:
        response = chain.invoke({"user_query": user_query})
        return json.loads(response.content.strip())
    except json.JSONDecodeError as e:
        logger.error(f"LLM did not return a valid JSON object for ensemble_model parameters: {e}")
        return None
    except Exception as e:
        logger.error(f"LLM call for ensemble_model parameters failed: {e}")
        return None

@tool("ensemble_model_tool", return_direct=True) 
def ensemble_model_tool(state: MLState, user_query: Optional[str] = None) -> Dict[str, Any]:
    """
    Creates an ensemble of a given model using bagging or boosting.

    This tool intelligently selects the best available model from the state and
    applies an ensembling method. It can parse natural language to determine the
    method ('bagging' or 'boosting') and the number of estimators.

    This tool is currently applicable only for classification and regression tasks.

    Args:
        state (MLState): The current state of the ML pipeline, which must include
                         a model to be ensembled.
        user_query (Optional[str]): The user's command with ensembling preferences.

    Returns:
        Dict[str, Any]: A dictionary containing the `ensemble_model`, an updated main `model`
                        reference, and the `last_output` with performance metrics to update the state.
    """
    logger.info("--- Executing Ensemble Model Tool ---")

    model_to_ensemble = None
    if state.tuned_model:
        model_to_ensemble = state.tuned_model
    elif state.created_model:
        model_to_ensemble = state.created_model
    elif state.best_model and not isinstance(state.best_model, list):
        model_to_ensemble = state.best_model
    elif state.best_model and isinstance(state.best_model, list) and len(state.best_model) > 0:
        model_to_ensemble = state.best_model[0] 

    if not state.setup_done or model_to_ensemble is None:
        return {"last_output": "❌ Please run setup and create/compare a model before ensembling. No suitable model found in state."}
    
    task = state.task
    if task not in ("classification", "regression"):
        return {"last_output": f"❌ `ensemble_model` tool is not applicable for the '{task}' task. It is only for classification and regression."}

    config = DEFAULT_ENSEMBLE_CONFIG.copy()
    
    if user_query:
        logger.info(f"Parsing user query for ensemble_model parameters: '{user_query}'")
        llm_updates = _get_llm_ensemble_parameters(user_query)
        if llm_updates:

            if 'method' in llm_updates:
                method = llm_updates['method'].capitalize() 
                if method in ["Bagging", "Boosting"]:
                    config['method'] = method
                else:
                    logger.warning(f"Invalid ensemble method '{llm_updates['method']}' requested. Using default: {config['method']}.")
            
            if 'n_estimators' in llm_updates and isinstance(llm_updates['n_estimators'], int) and llm_updates['n_estimators'] > 0:
                config['n_estimators'] = llm_updates['n_estimators']
            else:
                if 'n_estimators' in llm_updates:
                    logger.warning(f"Invalid n_estimators '{llm_updates['n_estimators']}' requested. Using default: {config['n_estimators']}.")

            if 'choose_better' in llm_updates and isinstance(llm_updates['choose_better'], bool):
                config['choose_better'] = llm_updates['choose_better']
            else:
                if 'choose_better' in llm_updates:
                    logger.warning(f"Invalid choose_better '{llm_updates['choose_better']}' requested. Using default: {config['choose_better']}.")
            
            logger.info(f"Applied LLM-driven config updates: {llm_updates}")

    logger.info(f"Final ensemble_model config: {config}")

    ensemble_map = {"classification": classification_ensemble, "regression": regression_ensemble}
    pull_map = {"classification": classification_pull, "regression": regression_pull}

    try:
        ensemble_function = ensemble_map[task]
        ensembled_model = ensemble_function(model_to_ensemble, **config)
        ensemble_results = pull_map[task]()
        
        logger.info(f"Successfully created ensemble model with method '{config['method']}'.")
        
        return {
            "ensemble_model": ensembled_model,
            "model": ensembled_model, # Update the main 'model' reference
            "last_output": f"✅ Ensemble model created with method '{config['method']}'.\n\n**Performance Metrics:**\n```\n{ensemble_results.to_string()}\n```"
        }
    except Exception as e:
        logger.error(f"PyCaret ensemble_model failed: {e}", exc_info=True)
        return {"last_output": f"❌ Failed to create ensemble model: {e}. Please check the model or ensembling parameters."}