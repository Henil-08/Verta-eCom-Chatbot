import os
import json
import uuid
import pandas as pd
from pandas import DataFrame
from pathlib import Path
from datetime import datetime
from sqlalchemy import text

from langfuse import Langfuse
from fastapi.responses import JSONResponse
from langchain_community.document_loaders import DataFrameLoader
from langchain_huggingface import HuggingFaceEmbeddings
from fastapi import HTTPException
from langchain_community.vectorstores import FAISS

from pydantic_models.models import scoreTrace
from utils.database import connect_with_db
from entity.config_entity import PrepareBaseModelConfig
from logger import logger

class Generate:
    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config
        self.vector_store_cache = []
        self.cache_creation_times = {}
        self.langfuse = Langfuse()


    def load_test_data(self):
        parquet_files = [f for f in os.listdir(self.config.testset_path) if f.endswith('.parquet')]

        with open(self.config.file_hash, 'r') as json_file:
            asin_uuid_map = json.load(json_file)

        uuid_asin_map = {v: k for k, v in asin_uuid_map.items()}

        dfs = [self.process_parquet(f, uuid_asin_map) for f in parquet_files]
        test_df = pd.concat(dfs, ignore_index=True)

        return test_df

    async def load_product_data(self, asin: str):
        credentials = {
            'INSTANCE_CONNECTION_NAME': os.getenv("INSTANCE_CONNECTION_NAME"),
            'DB_USER': os.getenv("DB_USER"),
            'DB_PASS': os.getenv("DB_PASS"),
            'DB_NAME': os.getenv("DB_NAME")
        }
        self.engine = connect_with_db(credentials=credentials)

        with self.engine.begin() as connection:
            try:
                # Log start of data fetching
                logger.info(f"Loading product data for ASIN: {asin}")

                # Fetch reviews
                review_query = text(f"""
                    SELECT parent_asin, asin, helpful_vote, timestamp, verified_purchase, title, text
                    FROM userreviews ur 
                    WHERE ur.parent_asin = '{asin}';
                """)
                review_result = connection.execute(review_query)
                review_df = pd.DataFrame(review_result.fetchall(), columns=review_result.keys())
                logger.info("Fetched review data")

                # Fetch metadata
                meta_query = text(f"""
                    SELECT parent_asin, main_category, title, average_rating, rating_number, features, description, price, store, categories, details
                    FROM metadata md 
                    WHERE md.parent_asin = '{asin}';
                """)
                meta_result = connection.execute(meta_query)
                meta_df = pd.DataFrame(meta_result.fetchall(), columns=meta_result.keys())
                logger.info("Fetched metadata")

            except Exception as e:
                logger.error(f"Error loading data for ASIN: {asin} - {e}")
                raise HTTPException(status_code=500, detail="Error loading data")

        return review_df, meta_df


    def create_vector_store(self, review_df: DataFrame):
        logger.info("Creating vector store from review data")
        review_df = review_df[review_df['text'].notna()]
        loader = DataFrameLoader(review_df)
        review_docs = loader.load()

        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectordb = FAISS.from_documents(documents=review_docs, embedding=embeddings)
        logger.info("Vector store created successfully")
        return vectordb


    async def initialize(self, asin: str, user_id: str, returnPath=False):
        cache_key = f"{user_id}-{asin}"
        retriever_path = Path(f"{self.config.faiss_dir}/{cache_key}")
        metadata_path = Path(f"{self.config.meta_dir}/{cache_key}.csv")

        if cache_key not in self.vector_store_cache:
            review_df, meta_df = await self.load_product_data(asin)
            vector_db = self.create_vector_store(review_df)

            # saving cache
            self.save_db(retriever_path, vector_db)
            self.save_csv(metadata_path, meta_df)

            self.vector_store_cache.append(cache_key)
            self.cache_creation_times[cache_key] = datetime.now()
            logger.info(f"Retriever initialized and cached for ASIN: {asin} and User ID: {user_id}")
        else:
            logger.info(f"Retriever for ASIN: {asin} and User ID: {user_id} already cached")

        if returnPath:
            return retriever_path, metadata_path


    async def score_feedback(self, score: scoreTrace):
        trace_id = score.run_id
        user_id = score.user_id
        asin = score.parent_asin
        value = score.value
        id = str(uuid.uuid4()) + f"-{user_id}-{asin}"
        
        
        self.langfuse.score(
            id=id,
            trace_id=trace_id,
            name="user-feedback",
            value=value,
            data_type="BOOLEAN" 
        )
        logger.info(f"Feedback Successful, 'trace_id': {trace_id}, 'id': {id}")
        
        return JSONResponse(content={"status": "Feedback Successful", "trace_id": trace_id}, status_code=200)


    @staticmethod
    def save_db(path: Path, vector_db):
        vector_db.save_local(path)
        logger.info(f"VectorDB saved at: {path}")


    @staticmethod
    def save_csv(path: Path, data: DataFrame):
        data.to_csv(path, index=False)
        logger.info(f"Metadata saved at: {path}")