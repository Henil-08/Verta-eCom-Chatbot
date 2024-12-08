from main.graph import Graph
from config.configuration import ConfigurationManager
from langgraph.graph.state import CompiledStateGraph

from logger import logger

import os
from dotenv import load_dotenv
load_dotenv()

## load the API Keys
os.environ['HF_TOKEN']=os.getenv("HF_TOKEN")
os.environ['OPENAI_API_KEY']=os.getenv("OPENAI_API_KEY")
os.environ['GROQ_API_KEY']=os.getenv("GROQ_API_KEY")
os.environ['LANGFUSE_PUBLIC_KEY']=os.getenv("LANGFUSE_PUBLIC_KEY")
os.environ['LANGFUSE_SECRET_KEY']=os.getenv("LANGFUSE_SECRET_KEY")
os.environ['LANGFUSE_HOST']=os.getenv("LANGFUSE_HOST")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

STAGE_NAME = "Create LangGraph Workflow"

class PrepareBaseTrainingPipeline:
    def __init__(self):
        pass

    def graph(self, isMemory=False) -> CompiledStateGraph:
        config = ConfigurationManager()
        prepare_base_model_config = config.get_prepare_base_model_config()
        prepare_base_model = Graph(config=prepare_base_model_config)
        app = prepare_base_model.create_graph(isMemory=isMemory)
        return app


if __name__ == "__main__":
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = PrepareBaseTrainingPipeline()
        app = obj.graph()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\n")
    except Exception as e:
        logger.exception(e)
        raise e