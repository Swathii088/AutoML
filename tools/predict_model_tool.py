import logging
import pandas as pd
from typing import Dict, Any, Optional
from pycaret.classification import predict_model as classification_predict
from pycaret.regression import predict_model as regression_predict
from pycaret.clustering import predict_model as clustering_predict
from pycaret.anomaly import predict_model as anomaly_predict
from langchain_core.tools import tool
from state.graph_state import MLState
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

def _get_active_model(state: MLState) -> Any | None:
    """Intelligently selects the best available model from the state."""
    # Prioritize in order: final > ensemble > tuned > best > created
    model = (
        state.final_model or
        state.ensemble_model or
        state.tuned_model or
        state.best_model or
        state.created_model
    )
    
    # If best_model is a list (from compare_models n_select > 1), take the first one
    if isinstance(model, list) and len(model) > 0:
        return model[0]
    return model

@tool("predict_model_tool", return_direct=True)
def predict_model_tool(state: MLState, new_data: pd.DataFrame) -> Dict[str, Any]:
    """
    Generates predictions on a new, unseen DataFrame using the active model.

    This tool is intended to be called by an agent when the user provides
    a new dataset (e.g., via upload) for prediction.

    Args:
        state (MLState): The current state of the ML pipeline, containing an active model.
        new_data (pd.DataFrame): The new, unseen data for which to generate predictions.
                                 This DataFrame is expected to be passed directly to the tool.

    Returns:
        Dict[str, Any]: A dictionary containing the `predictions` DataFrame and the
                        `last_output` message to update the state.
    """
    logger.info("--- Executing Predict Model Tool ---")
    model = _get_active_model(state)
    if not model:
        return {"last_output": "❌ No trained model found to make predictions. Please train a model first."}

    if not isinstance(new_data, pd.DataFrame) or new_data.empty:
        return {"last_output": "❌ No valid new data provided for prediction. Please ensure a DataFrame is passed."}

    if state.data is not None and isinstance(state.data, pd.DataFrame):
        original_cols = set(state.data.drop(columns=[state.target_column]).columns if state.target_column else state.data.columns)
        new_data_cols = set(new_data.columns)
        
        missing_cols = original_cols - new_data_cols
        extra_cols = new_data_cols - original_cols
        
        if missing_cols:
            logger.warning(f"New data is missing columns present in training data: {list(missing_cols)}")
            
        if extra_cols:
            logger.warning(f"New data has extra columns not present in training data: {list(extra_cols)}")

    try:
        predictions = None
        
        if state.task == "classification":
            predict_func = classification_predict
            predictions = predict_func(model, data=new_data, verbose=False)
        elif state.task == "regression":
            predict_func = regression_predict
            predictions = predict_func(model, data=new_data, verbose=False)
        elif state.task == "clustering":
            predict_func = clustering_predict 
            predictions = predict_func(model, data=new_data, verbose=False)
        elif state.task == "anomaly":
            predict_func = anomaly_predict
            predictions = predict_func(model, data=new_data, verbose=False)
        else:
            return {"last_output": f"❌ Prediction not supported for task type: {state.task}. Only classification, regression, clustering, and anomaly are supported."}

        if predictions is None:
             return {"last_output": f"❌ Prediction failed: No predictions generated for task type: {state.task}."}

        logger.info(f"Successfully generated predictions on new data.")
        
        return {
            "predictions": predictions, 
            "last_output": f"✅ Predictions generated successfully on the new data.\n\n**Prediction Preview (first 5 rows):**\n```\n{predictions.head().to_string()}\n```"
        }
    except Exception as e:
        logger.error(f"Prediction on new data failed: {e}", exc_info=True)
        return {"last_output": f"❌ Prediction failed on new data: {e}. Please ensure the new data format is compatible with the trained model."}

