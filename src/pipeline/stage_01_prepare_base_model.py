from src.graph import Graph
from src.config.configuration import ConfigurationManager
from langgraph.graph.state import CompiledStateGraph

from src import logger

STAGE_NAME = "Create LangGraph Workflow"


class PrepareBaseTrainingPipeline:
    def __init__(self):
        pass

    def graph(self, isMemory=True) -> CompiledStateGraph:
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