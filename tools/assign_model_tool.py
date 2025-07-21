import logging
from typing import Dict, Any
from pycaret.clustering import assign_model as clustering_assign
from pycaret.anomaly import assign_model as anomaly_assign
from langchain_core.tools import tool
from state.graph_state import MLState

logger = logging.getLogger(__name__)

def _get_active_model(state: MLState) -> Any | None:
    """Intelligently selects the best available model from the state."""
    model = (
        state.final_model or
        state.ensemble_model or
        state.tuned_model or
        state.best_model or
        state.created_model
    )
    if isinstance(model, list) and len(model) > 0:
        return model[0]
    return model

@tool("assign_model_tool")
def assign_model_tool_logic(state: MLState) -> Dict[str, Any]:
    """
    Applies an unsupervised model to the original dataset to generate labels or scores.

    This tool is used for clustering and anomaly detection to see the results
    (e.g., cluster labels, anomaly flags) on the training data.

    Args:
        state (MLState): The current state of the ML pipeline.

    Returns:
        Dict[str, Any]: A dictionary containing the updated `data` DataFrame
                        and a `last_output` message.
    """
    logger.info("--- Executing Assign Model Tool ---")
    model = _get_active_model(state)
    if not model:
        return {"last_output": "❌ No trained model found. Please train a model first."}

    task = state.task
    if task not in ("clustering", "anomaly"):
        return {"last_output": f"❌ `assign_model` is only for clustering and anomaly tasks."}

    try:
        if task == "clustering":
            data_with_labels = clustering_assign(model, verbose=False)
        else: 
            data_with_labels = anomaly_assign(model, verbose=False)
        
        logger.info(f"Successfully assigned model for {task} task.")
        
        return {
            "data": data_with_labels, 
            "last_output": f"✅ Model assigned successfully. New columns were added to the data.\n\n**Data Preview:**\n```\n{data_with_labels.head().to_string()}\n```"
        }
    except Exception as e:
        logger.error(f"Assign model failed: {e}", exc_info=True)
        return {"last_output": f"❌ Failed to assign model: {e}"}