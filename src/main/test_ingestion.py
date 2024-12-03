import os
import uuid
from logger import logger
from ragas.testset.evolutions import (simple, reasoning, multi_context)
from ragas.testset import TestsetGenerator


import pandas as pd
from pathlib import Path
from sqlalchemy import text
from langchain_community.document_loaders import DataFrameLoader
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from entity.config_entity import TestIngestionConfig
from utils.common import save_json, save_parquet
from utils.database import connect_with_db
from langchain.schema import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


class TestIngestion:
    def __init__(self, config: TestIngestionConfig, product_asins: list):
        self.config = config
        self.product_asins = product_asins
        self.hash_table = {asin: str(uuid.uuid4()) for asin in self.product_asins}


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
                logger.error(f"Unable to load data for asin {asin} Exception: {e}")

        return review_df, meta_df
    
    def generate_meta_summary(self, meta_df, prompt):
        meta_llm = ChatGroq(model_name="llama-3.1-8b-instant")

        modified_details = meta_df['details'].astype(str).str.replace('{', '[')
        
        # Answer question
        meta_system_prompt =( 
            prompt.format(
                main_category=(meta_df.at[0, 'main_category']),
                title=(meta_df.at[0, 'title']),
                average_rating=(meta_df.at[0, 'average_rating']),
                rating_number=(meta_df.at[0, 'rating_number']),
                features=(meta_df.at[0, 'features']),
                description=(meta_df.at[0, 'description']),
                price=(meta_df.at[0, 'price']),
                store=(meta_df.at[0, 'store']),
                categories=(meta_df.at[0, 'categories']),
                details=(modified_details.at[0]),
            )
        )

        meta_system_prompt = meta_system_prompt.replace('{', '{{').replace('}', '}}')

        meta_qa_prompt = ChatPromptTemplate.from_messages(
                        [
                            ("system", meta_system_prompt),
                        ]
                    )
        parser = StrOutputParser()
        meta_chain = meta_qa_prompt | meta_llm | parser

        try:
            # Meta Summary
            meta_results = meta_chain.invoke({'input': ''})
            meta_results = Document(page_content=meta_results, metadata={"source": "Metadata"})
            
        except Exception as error:
            content = "Metadata: Unable to generate result"
            meta_results = Document(page_content=content, metadata={"source": "Metadata"})

        return meta_results

    def create_docs(self, review_df, meta_df):
        review_df = review_df[review_df['text'].notna()]
        loader = DataFrameLoader(review_df)
        docs = loader.load()
        docs.insert(0, self.generate_meta_summary(meta_df, self.config.prompt_metadata))
        return docs

    def get_ragas_testset(self, doc, with_debugging_logs=False):    
        # generator with openai models
        generator_llm = ChatOpenAI(model="gpt-4o-mini")
        critic_llm = ChatOpenAI(model="gpt-4o")
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

        generator = TestsetGenerator.from_langchain(
            generator_llm,
            critic_llm,
            embeddings,
        )
        
        # generate testset
        testset = generator.generate_with_langchain_docs(
            doc, 
            test_size= 4,
            distributions={simple: 0.5, reasoning: 0.25, multi_context: 0.25},
            with_debugging_logs = with_debugging_logs,
        )
        return testset
    
    def transform_ragas_testset_df(self, testset, asin):
        ragas_test_df = testset.to_pandas()
        ragas_test_df['parent_asin'] = asin
        ragas_test_df.drop(columns=['metadata', 'episode_done'], inplace=True)
        return ragas_test_df
    
    def save_hash(self, hash):
        save_json(path=Path(f"{self.config.artifact_dir}/product_uuids.json"), data=hash)

    def generate_test_set(self):
        self.hash_table = {asin: str(uuid.uuid4()) for asin in self.product_asins}
        
        for idx, asin in enumerate(self.product_asins):
            review_df, meta_df = self.load_product_data(asin)

            # Load the Docs
            docs = self.create_docs(review_df, meta_df)

            try:
                logger.info(f"Generating Test Data for {asin}...")
                testset = self.get_ragas_testset(docs[:20] if len(docs) > 20 else docs)
                test_df = self.transform_ragas_testset_df(testset, asin)
                save_parquet(path=Path(f'{self.config.testset_path}/{self.hash_table[asin]}.parquet'), data=test_df)
            except Exception as e:
                logger.error(f'Error occurred while processing {asin}: {e}')
                continue
        
        self.save_hash(self.hash_table)