import os

from logger import logger
from main.failure_detection import FailureDetection

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
os.environ["MS_TEAMS_WEBHOOK_URL"]=os.getenv("MS_TEAMS_WEBHOOK_URL")


STAGE_NAME = "Failure Detection"

class FailureDetectionPipeline:
    def __init__(self):
        pass

    def detect_failure(self):
        detection = FailureDetection()
        detection.detect()
        

if __name__ == "__main__":
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        failure_detection = FailureDetectionPipeline()
        failure_detection.detect_failure()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\n")
    except Exception as e:
        logger.exception(e)
        raise e

