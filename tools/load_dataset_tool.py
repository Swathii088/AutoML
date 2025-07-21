import logging
import json
from typing import Dict, Any
from pycaret.datasets import get_data
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from state.graph_state import MLState # Adjust import path
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# Initialize the LLM with JSON mode enabled
llm = ChatGroq(
    temperature=0.0,
    model_name="llama3-70b-8192",
    model_kwargs={"response_format": {"type": "json_object"}},
)

AVAILABLE_DATASETS = [
    "airquality", "amazon", "anomaly", "asia_gdp", "automobile", "airline",
    "bank", "bike", "blood", "boston", "cancer", "concrete",
    "country-data", "credit", "delaware_anomaly", "diabetes",
    "diamond", "elections", "electrical_grid", "employee",
    "employee_performance", "energy", "facebook", "forest",
    "france", "germany", "glass", "gold", "heart",
    "heart_disease", "hepatitis", "house", "income", "index",
    "insurance", "ipl", "iris", "jewellery", "juice", "kiva",
    "mice", "migration", "nba", "parkinsons", "perfume",
    "pokemon", "poker", "population", "public_health",
    "pycaret_downloads", "questions", "satellite", "seeds",
    "spam", "spx", "telescope", "titanic", "traffic", "tweets",
    "us_presidential_election_results", "wholesale", "wikipedia",
    "wine"
]

def _get_llm_dataset_name(user_query: str) -> Dict[str, Any] | None:
    """Uses an LLM in JSON mode to extract the dataset name from the user's query."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an assistant that extracts information from the user's query. You must only respond with a valid JSON object."),
        ("user", "From the following query, extract the name of the dataset the user wants to load. The name must be one of the allowed options.\n\nAllowed Datasets: {dataset_options}\n\nUser Query: {user_query}\n\nJSON Output:")
    ])
    chain = prompt | llm
    try:
        response = chain.invoke({
            "dataset_options": AVAILABLE_DATASETS,
            "user_query": user_query
        })
        return json.loads(response.content)
    except Exception as e:
        logger.error(f"LLM call for dataset name failed: {e}")
        return None

def list_available_datasets() -> str:
    """Returns a formatted string of all available datasets."""
    return "\n".join([f"{i+1}. {name}" for i, name in enumerate(AVAILABLE_DATASETS)])

# Corrected function name for consistency
@tool("load_dataset_tool")
def load_dataset_tool(state: MLState, user_query: str) -> Dict[str, Any]:
    """
    Loads a dataset by name from PyCaret's collection into the state.
    It uses an LLM to identify the dataset name from the user's query.
    """
    logger.info("--- Executing Load Dataset Tool ---")
    
    params = _get_llm_dataset_name(user_query)
    if not params or not params.get("dataset_name"):
        return {"last_output": "❌ I could not identify a valid dataset name in your request."}

    dataset_name = params["dataset_name"].lower()

    if dataset_name not in AVAILABLE_DATASETS:
        return {"last_output": f"❌ Dataset '{dataset_name}' not found."}
    
    try:
        data = get_data(dataset_name, verbose=True)
        logger.info(f"Successfully loaded dataset '{dataset_name}'.")
        
        summary = f"✅ Dataset '{dataset_name}' loaded successfully! Shape: {data.shape}\n\n**Data Preview :**\n{data.head().to_dict}"

        return {
            "data": data,
            "last_output": summary
        }
    except Exception as e:
        logger.error(f"Failed to load dataset '{dataset_name}': {e}", exc_info=True)
        return {"last_output": f"❌ Error loading dataset '{dataset_name}': {e}"}