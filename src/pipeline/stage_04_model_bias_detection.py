# from src import logger
from typing import List, Dict
from langfuse import Langfuse
# import mlflow
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
import pandas as pd
from collections import Counter
import numpy as np
import os
from dotenv import load_dotenv
import torch
load_dotenv()

extended_review_df = pd.DataFrame({
    "asin": [
        "B00000IV35", "B00000IV35", "B00000IV35", "B00000IV35", "B00000IV35",
        "0975277324", "0975277324", "0975277324",
        "8499000606", "8499000606", "8499000606", "8499000606", "8499000606",
        "B00000IZJB", "B00000IZJB", "B00000IZJB", "B00000IZJB", "B00000IZJB",
        "1933054395", "1933054395", "1933054395", "1933054395",
        "B00005O6B7", "B00005O6B7", "B00005O6B7", "B00005O6B7"
    ],
    "text": [
        "This toy is amazing! My kids love it.",
        "Durable and well-designed, great value!",
        "Hours of fun, definitely recommend it.",
        "Bright colors and sturdy build, love it.",
        "Perfect for young children, a must-buy.",
        "An average product. Not worth the hype.",
        "Extremely bad quality. Do not buy this.",
        "Falls apart quickly, very disappointing.",
        "Fantastic game, enjoyed by the whole family.",
        "Not suitable for younger kids, too complicated.",
        "Average gameplay, gets boring quickly.",
        "Good concept but poorly executed.",
        "Family had a good time playing it.",
        "Very poor build quality, disappointing.",
        "Fell apart after a week of use, not worth it.",
        "Bad packaging and overpriced.",
        "Exceeded expectations, worth every penny.",
        "Good product but slightly overpriced.",
        "Great story, kept my kids engaged.",
        "Educational and fun, highly recommend.",
        "Perfect for young learners.",
        "A wonderful book, my children love it.",
        "Flimsy and poorly made, broke easily.",
        "Unimpressive quality, not worth the price.",
        "Not as described, false advertising.",
        "Could have been better with proper materials."
    ],
})

varied_evaluation_results = [
    # B00000IV35
    {
        "query": "Would you recommend this product?",
        "response": "The reviews for this product are overwhelmingly positive. Customers appreciate its durable design, vibrant colors, and suitability for young children. Users frequently mention that it provides hours of fun and is a great value for the price. Based on this feedback, I would recommend this product for families with young children.",
        "asin": "B00000IV35"
    },
    {
        "query": "What are the main highlights of this product?",
        "response": "Key highlights of this product include its bright colors, sturdy construction, and engaging design that ensures hours of entertainment for kids. Many users also praised its excellent value for money and its appeal to young children.",
        "asin": "B00000IV35"
    },
    {
        "query": "Does this product have any downsides?",
        "response": "While the majority of reviews are positive, a small number of users mentioned concerns about durability under heavy use. However, these instances appear to be rare and do not detract significantly from the overall user satisfaction.",
        "asin": "B00000IV35"
    },
    # 0975277324
    {
        "query": "Would you recommend this product?",
        "response": "The reviews for this product are mostly negative, with users highlighting concerns about its quality and durability. Many users reported that it broke quickly and failed to meet expectations. Based on the available feedback, I would not recommend this product.",
        "asin": "0975277324"
    },
    {
        "query": "What are the main highlights of this product?",
        "response": "Users struggled to find significant highlights for this product, as the feedback largely focuses on its shortcomings. It is described as average at best, with no outstanding features that set it apart.",
        "asin": "0975277324"
    },
    {
        "query": "Does this product have any downsides?",
        "response": "Yes, the most commonly reported downsides are poor quality and lack of durability. Many users expressed disappointment with the product, mentioning that it falls apart quickly and does not live up to its description.",
        "asin": "0975277324"
    },
    # 8499000606
    {
        "query": "Would you recommend this product?",
        "response": "The reviews for this product are mixed. While some users enjoyed it as a family game and appreciated its concept, others found it too complicated for younger children and felt the gameplay became repetitive over time. Whether to recommend it would depend on your intended audience.",
        "asin": "8499000606"
    },
    {
        "query": "What are the main highlights of this product?",
        "response": "This product is well-regarded for being enjoyable for families and offering an engaging gameplay experience. However, it may require some guidance for younger players due to its complexity.",
        "asin": "8499000606"
    },
    {
        "query": "Does this product have any downsides?",
        "response": "Some users noted that the game could be too complicated for younger players and that the gameplay might become repetitive after extended use. Additionally, a few users mentioned that the overall execution of the concept could be improved.",
        "asin": "8499000606"
    },
    # B00000IZJB
    {
        "query": "Would you recommend this product?",
        "response": "The reviews for this product are mostly negative. Customers frequently cited issues with its build quality, packaging, and overall value for money. Based on this feedback, I would not recommend this product.",
        "asin": "B00000IZJB"
    },
    {
        "query": "What are the main highlights of this product?",
        "response": "While a few users appreciated its functionality, most reviews did not identify significant highlights. Positive feedback is limited to isolated cases of meeting basic expectations.",
        "asin": "B00000IZJB"
    },
    {
        "query": "Does this product have any downsides?",
        "response": "Yes, common downsides include poor build quality, fragile components, and a lack of value for money. Several users reported that it broke after minimal use.",
        "asin": "B00000IZJB"
    },
    # 1933054395
    {
        "query": "Would you recommend this product?",
        "response": "The reviews for this product are overwhelmingly positive. Customers praised its educational content, engaging story, and suitability for young learners. Based on this feedback, I would highly recommend this product.",
        "asin": "1933054395"
    },
    {
        "query": "What are the main highlights of this product?",
        "response": "Highlights include its engaging and educational story, which captivates young readers. Many users appreciated its ability to combine learning with entertainment, making it a valuable addition to any child’s library.",
        "asin": "1933054395"
    },
    {
        "query": "Does this product have any downsides?",
        "response": "The reviews for this product do not mention significant downsides. It appears to be well-received across the board.",
        "asin": "1933054395"
    },
    # B00005O6B7
    {
        "query": "Would you recommend this product?",
        "response": "The reviews for this product highlight several issues with its build quality and overall durability. Based on this feedback, I would not recommend this product.",
        "asin": "B00005O6B7"
    },
    {
        "query": "What are the main highlights of this product?",
        "response": "Users did not highlight many positive features. The most notable feedback was related to its potential if the build quality were improved.",
        "asin": "B00005O6B7"
    },
    {
        "query": "Does this product have any downsides?",
        "response": "Yes, the most common complaints include flimsy construction, poor durability, and a lack of alignment with its product description.",
        "asin": "B00005O6B7"
    }
]

SPARSE_DATA_PHRASES = [
    "few reviews", 
    "limited data", 
    "insufficient information", 
    "not enough reviews", 
    "small sample size"
]



class BiasDetectionPipeline:
    def __init__(self, evaluation_results: List[Dict], review_df: pd.DataFrame):
        """
        Initialize the bias detection pipeline with evaluation results and review dataframe.
        """
        self.evaluation_results = evaluation_results
        self.review_df = review_df
        self.bias_detector = BiasDetection()
        self.bias_results = {}

    def detect_bias(self):
        """
        Detect bias in the responses generated during evaluation.
        """
        # logger.info("Starting bias detection...")


        for result in self.evaluation_results:
            query = result["query"]
            response = result["response"]
            asin = result["asin"]
            reviews = self.review_df[self.review_df['asin'] == asin]['text'].tolist()
            review_sentiments = self.bias_detector.analyze_sentiments(reviews)
            bias_data = self.bias_detector.detect_bias(
                query=query,
                response=response,
                review_sentiments=review_sentiments,
                reviews=reviews,
                num_reviews=len(reviews),
            )
            print(bias_data)
            if asin not in self.bias_results:
                    self.bias_results[asin] = {
                        "queries": [],
                        "responses": [],
                        "bias_detected_count": 0,
                        "bias_types": set(),
                        "num_reviews": len(reviews),
                        "review_sentiments": review_sentiments,
                    }

            self.bias_results[asin]["queries"].append(query)
            self.bias_results[asin]["responses"].append(response)

            if bias_data["bias_detected"]:
                self.bias_results[asin]["bias_detected_count"] += 1
                self.bias_results[asin]["bias_types"].update(bias_data["bias_types"])
        # print(self.bias_results)
            # Log bias detection results to MLflow
        # self.log_to_mlflow(query, response, bias_data, asin)
                
    def log_bias_results_to_mlflow(self):
        # """
        # Log bias detection results for all ASINs to MLflow.
        # """
        # for asin, data in self.bias_results.items():
        #     mlflow.log_param(f"ASIN_{asin}_Num_Reviews", data["num_reviews"])
        #     mlflow.log_param(f"ASIN_{asin}_Review_Sentiments", dict(data["review_sentiments"]))
        #     mlflow.log_metric(f"ASIN_{asin}_Bias_Detected_Count", data["bias_detected_count"])
        #     mlflow.log_param(f"ASIN_{asin}_Bias_Types", ", ".join(data["bias_types"]) if data["bias_types"] else "None")
        #     mlflow.log_param(f"ASIN_{asin}_Queries", data["queries"])
        #     mlflow.log_param(f"ASIN_{asin}_Responses", data["responses"])
        pass
        


class BiasDetection:
    def __init__(self):
        self.sentiment_analyzer = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")
        self.similarity_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        # self.langfuse = Langfuse(
        #     api_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
        #     api_secret=os.getenv("LANGFUSE_SECRET_KEY"),
        # )

    def analyze_sentiments(self, texts: List[str]) -> Counter:
        sentiments = self.sentiment_analyzer(texts)
        return Counter([s["label"] for s in sentiments])
    
    # def analyze_sentiments_with_probs(self, texts: List[str]) -> Dict[str, List[float]]:
    #     sentiments = self.sentiment_analyzer(texts, return_all_scores=True)
    #     prob_distribution = {"POSITIVE": [], "NEGATIVE": []}
        
    #     for sentiment in sentiments:
    #         for score in sentiment:
    #             prob_distribution[score["label"]].append(score["score"])
        
    #     return prob_distribution

    def analyze_sentiments_with_probs(self, response: str) -> Dict[str, float]:
        raw_output = self.sentiment_analyzer(response, return_all_scores=True)
        # print("Raw Sentiment Output:", raw_output)
        LABEL_MAPPING = {"LABEL_0": "negative", "LABEL_1": "neutral", "LABEL_2": "positive"}


        sentiment_scores = {}
        for sentiment in raw_output[0]:
            mapped_label = LABEL_MAPPING.get(sentiment["label"], sentiment["label"].lower())
            sentiment_scores[mapped_label] = sentiment["score"]

        return sentiment_scores

    def sparse_data_acknowledged(self, response: str) -> bool:
        response_embedding = self.similarity_model.encode(response, convert_to_tensor=True)
        phrase_embeddings = self.similarity_model.encode(SPARSE_DATA_PHRASES, convert_to_tensor=True)

        similarities = util.pytorch_cos_sim(response_embedding, phrase_embeddings)
        max_similarity = torch.max(similarities).item()

        return max_similarity > 0.6

    def detect_bias(self, query: str, response: str, review_sentiments: Counter, reviews: List[str], num_reviews: int) -> Dict:
        response_probs = self.analyze_sentiments_with_probs(response)
        response_prob_neg = response_probs.get("negative", 0.0)
        response_prob_pos = response_probs.get("positive", 0.0)
        response_prob_neu = response_probs.get("neutral", 0.0)

        bias_flags = {"bias_detected": False, "bias_types": []}
        # print(response)

        if response_prob_neg > 0.7 and review_sentiments["NEGATIVE"] > review_sentiments["POSITIVE"]:
            bias_flags["bias_detected"] = True
            bias_flags["bias_types"].append("over_reliance_on_negative")
        # print(num_reviews)

        if num_reviews < 4 and not self.sparse_data_acknowledged(response):

            bias_flags["bias_detected"] = True
            bias_flags["bias_types"].append("missing_data_acknowledgment")

        return bias_flags

        # Semantic mismatch with reviews
        # if not self.is_response_consistent(response, reviews):
        #     bias_flags["bias_detected"] = True
        #     bias_flags["bias_types"].append("semantic_mismatch")

    def is_response_consistent(self, response: str, reviews: List[str]) -> bool:

        response_embedding = self.similarity_model.encode(response, convert_to_tensor=True)
        review_embeddings = self.similarity_model.encode(reviews, convert_to_tensor=True)

        similarities = util.pytorch_cos_sim(response_embedding, review_embeddings)
        max_similarity = torch.max(similarities).item()

        return max_similarity > 0.7

bias_pipeline = BiasDetectionPipeline(varied_evaluation_results, extended_review_df)
bias_pipeline.detect_bias()