from src.model_evaluation import Evaluation
from src.config.configuration import ConfigurationManager
from langgraph.graph.state import CompiledStateGraph
from src import logger


STAGE_NAME = "Model Evaluation"


class ModelEvaluationPipeline:
    def __init__(self, graph: CompiledStateGraph):
        self.app = graph

    def evaluate(self):
        config = ConfigurationManager()
        eval_config = config.get_evaluation_config()
        prepare_base_model_config = config.get_prepare_base_model_config()
        evaluation = Evaluation(config=eval_config, 
                                base_config=prepare_base_model_config, 
                                graph=self.app)
        evaluation.evaluation()
        evaluation.save_score()
        evaluation.log_into_mlflow()


if __name__ == "__main__":
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ModelEvaluationPipeline()
        obj.evaluate()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\n")
    except Exception as e:
        logger.exception(e)
        raise e