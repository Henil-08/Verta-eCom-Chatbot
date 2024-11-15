from src import logger
from src.pipeline.stage_01_prepare_base_model import PrepareBaseTrainingPipeline

import os
from dotenv import load_dotenv
load_dotenv()

## load the API Keys
os.environ['HF_TOKEN']=os.getenv("HF_TOKEN")
os.environ['OPENAI_API_KEY']=os.getenv("OPENAI_API_KEY")
os.environ['GROQ_API_KEY']=os.getenv("GROQ_API_KEY")

## Langfuse
os.environ['LANGFUSE_PUBLIC_KEY']=os.getenv("LANGFUSE_PUBLIC_KEY")
os.environ['LANGFUSE_SECRET_KEY']=os.getenv("LANGFUSE_SECRET_KEY")
os.environ['LANGFUSE_HOST']=os.getenv("LANGFUSE_HOST")

## Postgres DB
credentials = {
    'INSTANCE_CONNECTION_NAME': os.getenv("INSTANCE_CONNECTION_NAME"),
    'DB_USER': os.getenv("DB_USER"),
    'DB_PASS': os.getenv("DB_PASS"),
    'DB_NAME': os.getenv("DB_NAME")
}

STAGE_NAME = "Create LangGraph Workflow"

try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    prepare_base = PrepareBaseTrainingPipeline()
    app = prepare_base.graph(isMemory=False)
    response = app.invoke({'question': 'Hello!', 
                           "meta_data": '',
                           "retriever": ''})
    print(response['answer'].content)
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\n")
except Exception as e:
    logger.exception(e)
    raise e 