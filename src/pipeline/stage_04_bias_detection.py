import os

from logger import logger
from main.bias_detection import BiasDetection
from config.configuration import ConfigurationManager

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


STAGE_NAME = "Bias Detection"

class BiasDetectionPipeline:
    def __init__(self):
        pass

    def detect_bias(self):
        config = ConfigurationManager()
        bias_config = config.get_bias_detection_config()
        detection = BiasDetection(config=bias_config)
        detection.detect()
                
    def log_bias_results_to_mlflow(self):
        pass
        

if __name__ == "__main__":
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        bias_detection = BiasDetectionPipeline()
        bias_detection.detect_bias()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\n")
    except Exception as e:
        logger.exception(e)
        raise e