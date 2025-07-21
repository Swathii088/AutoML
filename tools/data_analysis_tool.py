import logging
import pandas as pd
import numpy as np
from typing import Dict, Any
from langchain_core.tools import tool
from state.graph_state import MLState

# --- Configure logging ---
logger = logging.getLogger(__name__)


# --- Tool 1: Descriptive Statistics ---
@tool("descriptive_statistics_tool")
def descriptive_statistics_tool(state: MLState) -> Dict[str, Any]:
    """
    Calculates and displays descriptive statistics for the loaded dataset.
    This includes count, mean, standard deviation, min, max, and quartiles for
    numeric columns and value counts for categorical columns.
    """
    logger.info("--- Executing Descriptive Statistics Tool ---")
    
    data = state.data
    if data is None:
        return {"last_output": "❌ No data loaded. Please load a dataset first."}

    try:
        numeric_summary = data.select_dtypes(include=np.number).describe().to_string()
        
        summary_str = f"### Data Shape\n{data.shape}\n\n"
        summary_str += f"### Numeric Summary\n```\n{numeric_summary}\n```\n\n"
        
        return { "last_output": f"**📊 Descriptive Statistics:**\n{summary_str}" }
        
    except Exception as e:
        logger.error(f"Failed to generate descriptive statistics: {e}", exc_info=True)
        return {"last_output": f"❌ An error occurred while generating statistics: {e}"}


# --- Tool 2: Missing Values Analysis ---
@tool("missing_values_tool")
def missing_values_tool(state: MLState) -> Dict[str, Any]:
    """
    Analyzes and reports the number and percentage of missing values for each
    column in the loaded dataset.
    """
    logger.info("--- Executing Missing Values Tool ---")
    
    data = state.data
    if data is None:
        return {"last_output": "❌ No data loaded. Please load a dataset first."}

    try:
        missing_counts = data.isnull().sum()
        missing_percentage = (missing_counts / len(data)) * 100
        
        missing_df = pd.DataFrame({
            'Missing Values': missing_counts,
            'Percentage (%)': missing_percentage
        })
        
        missing_df = missing_df[missing_df['Missing Values'] > 0]

        if missing_df.empty:
            return {"last_output": "✅ No missing values were found in the dataset."}
        
        return {
            "last_output": f"**🔍 Missing Values Analysis:**\n```\n{missing_df.to_string()}\n```"
        }
    except Exception as e:
        logger.error(f"Failed to analyze missing values: {e}", exc_info=True)
        return {"last_output": f"❌ An error occurred while analyzing missing values: {e}"}


# --- Tool 3: Correlation Analysis ---
@tool("correlation_analysis_tool")
def correlation_analysis_tool(state: MLState) -> Dict[str, Any]:
    """
    Computes the correlation matrix for numeric variables and identifies pairs
    with a high correlation coefficient (absolute value > 0.8).
    """
    logger.info("--- Executing Correlation Analysis Tool ---")
    
    data = state.data
    if data is None:
        return {"last_output": "❌ No data loaded. Please load a dataset first."}

    try:
        numeric_data = data.select_dtypes(include=np.number)
        if len(numeric_data.columns) < 2:
            return {"last_output": "Not enough numeric columns to compute correlations."}
            
        corr_matrix = numeric_data.corr()
        high_corrs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                if abs(corr_matrix.iloc[i, j]) > 0.8:
                    high_corrs.append(
                        f"- `{corr_matrix.columns[i]}` and `{corr_matrix.columns[j]}`: {corr_matrix.iloc[i, j]:.2f}"
                    )
        
        high_corr_summary = "\n".join(high_corrs) if high_corrs else "No highly correlated pairs found."
        
        full_report = f"""
### 🔗 High Correlation Pairs (>0.8)
{high_corr_summary}

---

### Full Correlation Matrix
{corr_matrix.to_string()}

"""
        return {"last_output": full_report}

    except Exception as e:
        logger.error(f"Failed to compute correlations: {e}", exc_info=True)
        return {"last_output": f"❌ An error occurred during correlation analysis: {e}"}