import os
import nest_asyncio
nest_asyncio.apply()

from logger import logger
from pipeline.stage_01_prepare_base_model import PrepareBaseTrainingPipeline
from pipeline.stage_02_test_data_ingestion import TestIngestionPipeline
from pipeline.stage_03_model_evaluation import ModelEvaluationPipeline
from pipeline.stage_04_bias_detection import BiasDetectionPipeline
from pipeline.stage_05_failure_detection import FailureDetectionPipeline

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
os.environ["MS_TEAMS_WEBHOOK_URL"]=os.getenv("MS_TEAMS_WEBHOOK_URL")


# STAGE_NAME = "Create LangGraph Workflow"
# try:
#     logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
#     prepare_base = PrepareBaseTrainingPipeline()
#     app = prepare_base.graph(isMemory=False)
#     response = app.invoke({'question': 'Hello!', 
#                            "meta_data": '',
#                            "retriever": ''})
#     if(response['answer'].content):
#         logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\n")
# except Exception as e:
#     logger.exception(e)
#     raise e 


# STAGE_NAME = "Test Data Ingestion"
# try:
#     logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
#     ingest = TestIngestionPipeline()
#     ingest.ingest()
#     logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\n")
# except Exception as e:
#     logger.exception(e)
#     raise e


# STAGE_NAME = "Model Evaluation"
# try:
#     logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
#     eval = ModelEvaluationPipeline(app)
#     eval.evaluate()
#     logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\n")
# except Exception as e:
#     logger.exception(e)
#     raise e


# STAGE_NAME = "Bias Detection"
# try:
#     logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
#     bias_detection = BiasDetectionPipeline()
#     bias_detection.detect_bias()
#     logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\n")
# except Exception as e:
#     logger.exception(e)
#     raise e


STAGE_NAME = "Failure Detection"
try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    failure_detection = FailureDetectionPipeline()
    failure_detection.detect_failure()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\n")
except Exception as e:
    logger.exception(e)
    raise e