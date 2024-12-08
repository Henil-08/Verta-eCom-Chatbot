from pathlib import Path

CONFIG_FILE_PATH = Path("config/config.yaml")
PROMPTS_FILE_PATH = Path("config/prompts.yaml")

# paths for Merging the Data
TOY_METADATA = Path("artifacts/meta_Toys_and_Games.jsonl")
VIDEO_GAME_METADATA = Path("artifacts/meta_Video_Games.jsonl")
TOY_REVIEW = Path("artifacts/review_Toys_and_Games.jsonl")
VIDEO_GAME_REVIEW = Path("artifacts/review_Video_Games.jsonl")
OUTPUT_META = Path('artifacts/meta_Toys_and_Video_Games.jsonl')
OUTPUT_REVIEW = Path('artifacts/review_Toys_and_Video_Games.jsonl')

# Agent Members
MEMBERS = ["Review-Vectorstore"]
OPTIONS = MEMBERS + ["FINISH"]

# Routing Function
CONDITIONAL_MAP = {k: k for k in MEMBERS}
CONDITIONAL_MAP["FINISH"] = 'generate'

# BigQuery Tables
REVIEW_TABLE = "ecom-chat-437005.ecom_chat.review"
META_TABLE = "ecom-chat-437005.ecom_chat.meta"
PARENT_ASIN = 'B072K6TLJX'

# Bias Detection Phrases
SPARSE_DATA_PHRASES = [
    "few reviews", 
    "limited data", 
    "insufficient information", 
    "not enough reviews", 
    "small sample size"
]

# Thresholds
THRESHOLDS = {
    "context_precision": 0.5,
    "faithfulness": 0.15,
    "answer_relevancy": 0.5,
    "context_recall": 0.5,
}