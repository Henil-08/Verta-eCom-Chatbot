import nest_asyncio
nest_asyncio.apply()

from main.test_ingestion import TestIngestion
from config.configuration import ConfigurationManager
from logger import logger

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


STAGE_NAME = "Test Data Ingestion"


class TestIngestionPipeline:
    def __init__(self):
        pass

    def ingest(self):
        product_asins = [
            "B00000IV35", "0975277324", "8499000606", "B00000IZJB", "1933054395", 
            "0976990709", "B00000IZKX", "B00000ISC5", "B00001ZWV7", "B00005O6B7", 
            "B0000205XI", "B00000DMD2", "B00000IV95", "B00000IV34", "B00005BZKD", 
            "1932855785", "B00000JBMZ", "B00004W3Y4", "B00004TFLB", "160169024X", 
            "B00000JIVS", "B00004YO15", "2914849656", "B00004NKLB", "B00000DMER", 
            "B000062SPJ", "B00000IZOU", "B00003008E", "076245945X", "B000050B3H"
        ]
    
        config = ConfigurationManager()
        test_ingest_config = config.get_test_ingestion_config()
        ingestion = TestIngestion(config=test_ingest_config, 
                                product_asins=product_asins)
        ingestion.generate_test_set()


if __name__ == "__main__":
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = TestIngestionPipeline()
        obj.ingest()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\n")
    except Exception as e:
        logger.exception(e)
        raise e