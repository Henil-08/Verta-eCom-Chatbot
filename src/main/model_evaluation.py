import os
import json
import time
import uuid
import mlflow
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_recall,
    context_precision,
)
from datasets import Dataset
from urllib.parse import urlparse
import pandas as pd
from pathlib import Path
from sqlalchemy import text
from langgraph.graph.state import CompiledStateGraph
from langfuse.callback import CallbackHandler
from langchain_community.document_loaders import DataFrameLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from urllib.parse import urlparse
from entity.config_entity import EvaluationConfig, PrepareBaseModelConfig
from utils.common import save_json, save_parquet
from utils.database import connect_with_db


class Evaluation:
    def __init__(self, config: EvaluationConfig, base_config: PrepareBaseModelConfig, graph: CompiledStateGraph):
        self.config = config
        self.base_config = base_config
        self.app = graph
        self.vector_store_cache = []
        self.results = pd.DataFrame()
        self.review_df = pd.DataFrame()
        self.test_df = pd.DataFrame()

    
    # Read and combine DataFrames
    def process_parquet(self, file, uuid_asin_map):
        df = pd.read_parquet(os.path.join(self.config.testset_path, file))
        df['file_hash'] = file.split('.')[0]
        df['parent_asin'] = df['file_hash'].map(uuid_asin_map)
        return df


    def load_test_data(self):
        parquet_files = [f for f in os.listdir(self.config.testset_path) if f.endswith('.parquet')]

        with open(self.config.file_hash, 'r') as json_file:
            asin_uuid_map = json.load(json_file)

        uuid_asin_map = {v: k for k, v in asin_uuid_map.items()}

        dfs = [self.process_parquet(f, uuid_asin_map) for f in parquet_files]
        test_df = pd.concat(dfs, ignore_index=True)

        return test_df[:30]

    def load_product_data(self, asin: str):
        credentials = {
            'INSTANCE_CONNECTION_NAME': os.getenv("INSTANCE_CONNECTION_NAME"),
            'DB_USER': os.getenv("DB_USER"),
            'DB_PASS': os.getenv("DB_PASS"),
            'DB_NAME': os.getenv("DB_NAME")
        }
        self.engine = connect_with_db(credentials=credentials)

        with self.engine.begin() as connection:
            try:
                # Fetch reviews
                review_query = text(f"""
                    SELECT parent_asin, asin, helpful_vote, timestamp, verified_purchase, title, text
                    FROM userreviews ur 
                    WHERE ur.parent_asin = '{asin}';
                """)
                review_result = connection.execute(review_query)
                review_df = pd.DataFrame(review_result.fetchall(), columns=review_result.keys())
                
                # Fetch metadata
                meta_query = text(f"""
                    SELECT parent_asin, main_category, title, average_rating, rating_number, features, description, price, store, categories, details
                    FROM metadata md 
                    WHERE md.parent_asin = '{asin}';
                """)
                meta_result = connection.execute(meta_query)
                meta_df = pd.DataFrame(meta_result.fetchall(), columns=meta_result.keys())
                
            except Exception as e:
                print("Exception: {}".format(e))

        return review_df, meta_df
    

    def create_vector_store(self, review_df):
        review_df = review_df[review_df['text'].notna()]
        self.review_df = review_df
        loader = DataFrameLoader(review_df)
        review_docs = loader.load()

        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectordb = FAISS.from_documents(documents=review_docs, embedding=embeddings)
        return vectordb


    def generate_response(self):
        test_df = self.load_test_data()
        
        for index, row in test_df.iterrows():
            cache_key = f"{row['file_hash']}-{row['parent_asin']}"
            
            if cache_key not in self.vector_store_cache:
                review_df, meta_df = self.load_product_data(row['parent_asin'])
                vector_db = self.create_vector_store(review_df)
                vector_db.save_local(f"{self.base_config.faiss_dir}/{cache_key}")
                meta_df.to_csv(f"{self.base_config.meta_dir}/{cache_key}.csv", index=False)
                self.vector_store_cache.append(cache_key)

            retriever = Path(f"{self.base_config.faiss_dir}/{cache_key}")
            meta_df = Path(f"{self.base_config.meta_dir}/{cache_key}.csv")

            lang_config = {}
            run_id = str(uuid.uuid4())
            langfuse_handler = CallbackHandler(
                user_id=f"Model-Evaluation-1",
                session_id=f"{cache_key}"
            )
            lang_config.update({"callbacks": [langfuse_handler], "run_id": run_id})


            try:
                response = self.app.invoke({
                    "question": row['question'], 
                    "meta_data": str(meta_df),
                    "retriever": str(retriever)
                }, config=lang_config)
                
                test_df.at[index, 'answer'] = response['answer'].content
                time.sleep(2)
            except Exception as e:
                row['answer']=''
                print(f"Error invoking agent for Index: {index} - {e}")
        
        return test_df

    def evaluation(self):
        test_df = self.generate_response()
        self.test_df = test_df
        test_dataset = Dataset.from_pandas(test_df)
        result = evaluate(
            test_dataset,
            metrics=[
                context_precision,
                faithfulness,
                answer_relevancy,
                context_recall,
            ],
        )
        results_df = result.to_pandas()
        columns_of_interest = ['context_precision', 'faithfulness', 'answer_relevancy', 'context_recall']
        self.results = results_df.groupby('evolution_type')[columns_of_interest].mean()

        self.save_results(Path(f'{self.config.results_path}/base.parquet'), test_df)
        self.save_results(Path(f'{self.config.results_path}/base-results.parquet'), results_df)
        
        
    def save_results(self, path, results):
        save_parquet(path=path, data=results)

    def save_score(self):
        metrics_dict = self.results.to_dict()
        save_json(path=Path(f"{self.config.metrics_path}/base-scores.json"), data=metrics_dict)

    def log_into_mlflow(self):
        mlflow.models.set_model(self.app)
        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        
        with mlflow.start_run():
            mlflow.log_params(self.config.all_params)

            for evolution_type, row in self.results.iterrows():
                for metric_name, metric_value in row.items():
                    metric_name_with_type = f"{evolution_type}_{metric_name}"
                    # Log the metric
                    mlflow.log_metric(metric_name_with_type, metric_value)

            # Model registry does not work with file store
            if tracking_url_type_store != "file":
                # Register the model
                mlflow.langchain.log_model(
                lc_model=self.config.lc_model,
                artifact_path=self.config.artifact_path,
                pip_requirements=self.config.pip_requirements,
                registered_model_name=self.config.registered_model_name
            )
            else:
                mlflow.langchain.log_model(
                lc_model=self.config.lc_model,
                artifact_path=self.config.artifact_path,
                pip_requirements=self.config.pip_requirements,
            )