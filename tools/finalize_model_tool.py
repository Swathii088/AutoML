import logging
from typing import Dict, Any, Optional
from pycaret.classification import finalize_model as classification_finalize
from pycaret.regression import finalize_model as regression_finalize
from langchain_core.tools import tool

from state.graph_state import MLState # Adjust import path as needed

logger = logging.getLogger(__name__)

@tool("finalize_model_tool", return_direct=True) # Added return_direct=True
def finalize_model_tool(state: MLState) -> Dict[str, Any]:
    """
    Finalizes the best available model by retraining it on the full dataset.

    This tool is the last training step before saving a model for production. It
    intelligently selects the most refined model from the state (ensemble > tuned > best > created)
    and trains it on 100% of the data to maximize performance.

    This tool is applicable only for classification and regression tasks.

    Args:
        state (MLState): The current state of the ML pipeline, which must include a
                         completed setup and at least one trained model.

    Returns:
        Dict[str, Any]: A dictionary containing the `final_model` object and the
                        `last_output` message to update the state.
    """
    logger.info("--- Executing Finalize Model Tool ---")

    if not state.setup_done:
        return {"last_output": "❌ Please run setup before finalizing a model."}

    # Intelligent model selection hierarchy
    model_to_finalize = (
        state.ensemble_model or
        state.tuned_model or
        state.created_model or
        state.best_model 
    )

    # Handle case where best_model might be a list (from compare_models with n_select > 1)
    if isinstance(model_to_finalize, list) and len(model_to_finalize) > 0:
        # If it's a list, we assume the first one is the intended model to finalize
        model_to_finalize = model_to_finalize[0]
        logger.info(f"Multiple models found in state. Selecting the first one for finalization: {type(model_to_finalize).__name__}")
    elif model_to_finalize is None:
        return {"last_output": "❌ No trained model found to finalize. Please create, compare, tune, or ensemble a model first."}

    task = state.task
    if task not in ("classification", "regression"):
        return {"last_output": f"❌ `finalize_model` tool is not applicable for the '{task}' task. It is only for classification and regression."}

    logger.info(f"Finalizing model '{type(model_to_finalize).__name__}' by training on the entire dataset...")
    
    finalize_map = {
        "classification": classification_finalize,
        "regression": regression_finalize
    }

    try:
        finalize_function = finalize_map[task]
        # PyCaret's finalize_model typically doesn't have a 'verbose' parameter,
        # but if it did, we'd set it to False for automated use.
        final_model = finalize_function(model_to_finalize)
        logger.info("Model finalized successfully.")

        # Get model details; PyCaret models often have a __str__ or __repr__
        model_details = str(final_model)
        
        return {
            "final_model": final_model,
            "model": final_model, # Update the main 'model' reference to the finalized one
            "last_output": f"✅ Model finalized successfully. It has been retrained on the full dataset and is ready for deployment.\n\n**Final Model Details:**\n```\n{model_details}\n```"
        }
    except Exception as e:
        logger.error(f"Failed to finalize model: {e}", exc_info=True)
        return {"last_output": f"❌ Failed to finalize model: {e}. Please ensure the model is valid for finalization."}

