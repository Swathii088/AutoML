import logging
import json
from typing import Dict, Any, Optional
from pycaret.classification import create_model as classification_create, pull as classification_pull
from pycaret.regression import create_model as regression_create, pull as regression_pull
from pycaret.clustering import create_model as clustering_create, pull as clustering_pull
from pycaret.anomaly import create_model as anomaly_create, pull as anomaly_pull
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from state.graph_state import MLState
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)
llm = ChatGroq(
    temperature=0.0,
    model_name="llama3-70b-8192",
    model_kwargs={"response_format": {"type": "json_object"}},
)

# --- Configuration ---

ALLOWED_ESTIMATORS = {
    "classification": ["lr", "knn", "nb", "dt", "svm", "rbfsvm", "gpc", "mlp", "ridge", "rf", "qda", "ada", "gbc", "lda", "et", "xgboost", "lightgbm", "catboost"],
    "regression": ["lr", "lasso", "ridge", "en", "lar", "llar", "omp", "br", "ard", "par", "ransac", "tr", "huber", "kr", "svm", "knn", "dt", "rf", "et", "ada", "gbr", "mlp", "xgboost", "lightgbm", "catboost"],
    "clustering": ["kmeans", "ap", "meanshift", "sc", "hclust", "dbscan", "optics", "birch"],
    "anomaly": ["iforest", "knn", "lof", "svm", "pca", "mcd", "sod", "sos"]
}


DEFAULT_SUPERVISED_CONFIG = { "fold": 10, "round": 4, "cross_validation": True }
DEFAULT_UNSUPERVISED_CONFIG = { "num_clusters": 4, "fraction": 0.05 } # Common defaults

def _get_llm_create_parameters(user_query: str, task: str) -> Dict[str, Any] | None:
    """Uses an LLM to parse the user query for create_model parameters."""

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "You are a machine learning assistant. Extract the `estimator` and any other relevant parameters (e.g., 'fold', 'num_clusters') from the user's query for PyCaret's `create_model()` function. Respond with a valid JSON object."),
        ("user", "Allowed estimators for task '{task}': {estimators}\n\nUser Query: {user_query}\n\nJSON Output:")
    ])
    chain = prompt_template | llm
    try:
        response = chain.invoke({
            "task": task,
            "estimators": ALLOWED_ESTIMATORS.get(task, []),
            "user_query": user_query
        })
        return json.loads(response.content.strip())
    except Exception as e:
        logger.error(f"LLM call for create_model parameters failed: {e}")
        return None
    

@tool("create_model_tool", return_direct=True) 
def create_model_tool(state: MLState, user_query: str) -> Dict[str, Any]:
    """
    Creates, trains, and evaluates a single model using PyCaret's `create_model`.

    This tool requires the user to specify a model estimator in the query (e.g., 
    'create a random forest model'). It uses an LLM to identify the model ID 
    and any other training parameters from the user's request.

    Args:
        state (MLState): The current state of the ML pipeline, which must include
                         a successfully run setup pipeline.
        user_query (str): The user's command specifying which model to create.

    Returns:
        Dict[str, Any]: A dictionary containing the `created_model` object and
                        the `last_output` with performance metrics to update the state.
    """
    logger.info("--- Executing Create Model Tool ---")

    if not state.setup_done:
        return {"last_output": "❌ Please run the setup tool before creating a model."}

    task = state.task
    if task not in ALLOWED_ESTIMATORS:
        return {"last_output": f"❌ The `create_model` tool is not applicable for the '{task}' task."}

    llm_updates = _get_llm_create_parameters(user_query, task)
    
    if not llm_updates or llm_updates.get("estimator") is None:
        return {"last_output": f"❌ Could not identify which model to create. Please specify one, e.g., 'create a random forest model'."}

    estimator = llm_updates.pop("estimator") 
    if estimator not in ALLOWED_ESTIMATORS[task]:
        return {"last_output": f"❌ Model '{estimator}' is not supported for a {task} task."}
    
    is_supervised = task in ["classification", "regression"]
    config = (DEFAULT_SUPERVISED_CONFIG if is_supervised else DEFAULT_UNSUPERVISED_CONFIG).copy()
    config.update(llm_updates)
    
    logger.info(f"Creating model '{estimator}' with config: {config}")

    create_map = {
        "classification": classification_create, "regression": regression_create,
        "clustering": clustering_create, "anomaly": anomaly_create
    }
    pull_map = {
        "classification": classification_pull, "regression": regression_pull,
        "clustering": clustering_pull, "anomaly": anomaly_pull
    }

    try:
        create_function = create_map[task]
        pull_function = pull_map[task]

        trained_model = create_function(estimator, **config, verbose=False)
        model_results = pull_function()
        
        logger.info(f"Successfully created model '{estimator}'.")
        
        return {
            "model": trained_model,
            "created_model": trained_model,
            "last_output": f"✅ Model '{estimator}' created successfully.\n\n**Results:**\n```\n{model_results.to_string()}\n```"
        }
    except Exception as e:
        logger.error(f"PyCaret create_model failed for '{estimator}': {e}", exc_info=True)
        return {"last_output": f"❌ Failed to create model '{estimator}': {e}."}