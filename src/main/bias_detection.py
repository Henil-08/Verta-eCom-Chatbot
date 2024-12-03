import os
import json
from sqlalchemy import text
import torch
import pandas as pd
from pathlib import Path

from typing import List, Dict
from collections import Counter
from langchain_huggingface import HuggingFaceEmbeddings

from logger import logger
from components.sentiments import sentiment_model, prob_sentiment_model
from constants import SPARSE_DATA_PHRASES
from entity.config_entity import BiasDetectionConfig
from utils.common import save_json, make_serializable
from utils.database import connect_with_db


class BiasDetection:
    def __init__(self, config: BiasDetectionConfig):
        self.config = config
        self.results = {}
        self.evaluation_results = self.read_parquet(path=self.config.results_path)

        try:
            self.embeddings = HuggingFaceEmbeddings(model_name=self.config.embedding_model)
        except:
            self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            logger.warning(f"Incorrect Embedding Model name {self.config.embedding_model}, using 'all-MiniLM-L6-v2'")
        
        self.sentiment_analyzer = sentiment_model(
            model_name=self.config.sentiment_model, 
            prompt=self.config.prompt_sentiment
        )
        
        self.prob_sentiment_analyzer = prob_sentiment_model(
            model_name=self.config.sentiment_model, 
            prompt=self.config.prompt_prob_sentiment
        )


    def read_parquet(self, path):
        df = pd.read_parquet(os.path.join(path, 'base-results.parquet'))
        return df


    def load_product_reviews(self, asin: str):
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
            except Exception as e:
                print("Exception: {}".format(e))

        review_df["text"] = review_df.apply(lambda row: f"title: {row['title']}\ncontent: {row['text']}", axis=1)

        return review_df[:20]


    def analyze_sentiments(self, texts: List[str]) -> Counter:
        sentiment_counts = Counter({"positive": 0, "neutral": 0, "negative": 0})
    
        for review in texts:
            try:
                # Send each review to the LLM for classification
                sentiment_response = self.sentiment_analyzer.invoke({"review": review})
                
                # Standardize the response to lowercase for counting
                sentiment = sentiment_response.lower()
                if 'positive' in sentiment:
                    sentiment_counts['positive'] += 1
                elif 'negative' in sentiment:
                    sentiment_counts['negative'] += 1
                elif 'neutral' in sentiment:
                    sentiment_counts['neutral'] += 1
                else:
                    print(f"Unexpected sentiment response: {sentiment_response}")
            except Exception as e:
                print(f"Error analyzing sentiment for review: {review}. Error: {e}")
        
        return sentiment_counts


    def analyze_sentiments_with_probs(self, response: str) -> Dict[str, float]:
        response = self.prob_sentiment_analyzer.invoke({'response': response})
        try:
            sentiment_probs = json.loads(response)
            return sentiment_probs
        except Exception as e:
            print(f"Error parsing sentiment probabilities: {e}")
            return {"positive": 0.0, "neutral": 0.0, "negative": 0.0}


    def sparse_data_acknowledged(self, response: str) -> bool:
        response_embedding = self.embeddings.embed_query(response)
        phrase_embeddings = [self.embeddings.embed_query(phrase) for phrase in SPARSE_DATA_PHRASES]
        similarities = torch.tensor([torch.cosine_similarity(torch.tensor(response_embedding), torch.tensor(embeddings), dim=0) for embeddings in phrase_embeddings])

        max_similarity = torch.max(similarities).item()

        return max_similarity > 0.6


    def bias_detection(self, response: str, review_sentiments: Counter, num_reviews: int) -> Dict:
        response_probs = self.analyze_sentiments_with_probs(response)
        response_prob_neg = response_probs.get("negative", 0.0)

        review_pos = review_sentiments.get("positive", 0)
        review_neg = review_sentiments.get("negative", 0)

        bias_flags = {"bias_detected": False, "bias_types": []}

        if response_prob_neg > 0.7 and review_neg > review_pos:
            bias_flags["bias_detected"] = True
            bias_flags["bias_types"].append("over_reliance_on_negative")

        if num_reviews < 4 and not self.sparse_data_acknowledged(response):

            bias_flags["bias_detected"] = True
            bias_flags["bias_types"].append("missing_data_acknowledgment")

        return bias_flags
    

    def save_score(self):
        save_json(path=Path(f"evaluation/metrics/bias-scores.json"), data=make_serializable(self.results))

    
    def detect(self):
        for index, row in self.evaluation_results.iterrows():
            response = row["answer"]
            asin = row["parent_asin"]
            logger.info(f"Detecting Bias for asin: {asin}")

            reviews = self.load_product_reviews(asin)
            review_sentiments = self.analyze_sentiments(reviews['text'].to_list())

            bias_data = self.bias_detection(
                response=response,
                review_sentiments=review_sentiments,
                num_reviews=len(reviews),
            )
            
            if asin not in self.results:
                self.results[asin] = {
                    "bias_detected_count": 0,
                    "bias_types": set(),
                    "num_reviews": len(reviews),
                    "review_sentiments": review_sentiments,
                }

            if bias_data["bias_detected"]:
                self.results[asin]["bias_detected_count"] += 1
                self.results[asin]["bias_types"].update(bias_data["bias_types"])

        self.save_score()