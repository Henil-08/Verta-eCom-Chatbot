from main.model_evaluation import Evaluation
from config.configuration import ConfigurationManager
from langgraph.graph.state import CompiledStateGraph
from logger import logger
from pipeline.stage_01_prepare_base_model import PrepareBaseTrainingPipeline

import os
from dotenv import load_dotenv
load_dotenv()

## load the API Keys
os.environ['HF_TOKEN']=os.getenv("HF_TOKEN")
os.environ['OPENAI_API_KEY']=os.getenv("OPENAI_API_KEY")
os.environ['GROQ_API_KEY']=os.getenv("GROQ_API_KEY")
os.environ['GOOGLE_APPLICATION_CREDENTIALS']=os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
os.environ['LANGFUSE_PUBLIC_KEY']=os.getenv("LANGFUSE_PUBLIC_KEY")
os.environ['LANGFUSE_SECRET_KEY']=os.getenv("LANGFUSE_SECRET_KEY")
os.environ['LANGFUSE_HOST']=os.getenv("LANGFUSE_HOST")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["MLFLOW_TRACKING_URI"]=os.getenv("MLFLOW_TRACKING_URI")
os.environ["MLFLOW_TRACKING_USERNAME"]=os.getenv("MLFLOW_TRACKING_USERNAME")
os.environ["MLFLOW_TRACKING_PASSWORD"]=os.getenv("MLFLOW_TRACKING_PASSWORD")


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
        prepare_base = PrepareBaseTrainingPipeline()
        app = prepare_base.graph(isMemory=False)
        eval = ModelEvaluationPipeline(app)
        eval.evaluate()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\n")
    except Exception as e:
        logger.exception(e)
        raise e