
from tools.setup_tool import setup_tool
from tools.create_model_tool import create_model_tool
from tools.tune_model_tool import tune_model_tool
from tools.ensemble_model_tool import ensemble_model_tool
from tools.automl_tool import automl_tool
from tools.leaderboard_tool import leaderboard_tool
from llm.task_identifier import identify_task_and_target
from tools.predict_model_tool import predict_model_tool
from tools.save_model_tool import save_model_tool
from tools.load_model_tool import load_model_tool

all_tools = [
    setup_tool,
    create_model_tool,
    tune_model_tool,
    ensemble_model_tool,
    identify_task_and_target,
    #automl_tool,
    leaderboard_tool,
    #predict_model_tool,
    save_model_tool,
    load_model_tool,
]
