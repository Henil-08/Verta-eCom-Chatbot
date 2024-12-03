from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class PrepareBaseModelConfig:
    cache_dir: Path
    faiss_dir: Path
    meta_dir: Path
    supervisor_model: str
    metadata_model: str
    base_model: str
    followup_model: str
    prompt_supervisor: str
    prompt_metadata: str
    prompt_base_model: str
    prompt_followup: str

@dataclass(frozen=True)
class TestIngestionConfig:
    artifact_dir: Path
    evaluation_root_dir: Path
    testset_path: Path
    prompt_metadata: str

@dataclass(frozen=True)
class EvaluationConfig:
    root_dir: Path
    metrics_path: Path
    results_path: Path
    testset_path: Path
    file_hash: Path
    lc_model: str
    artifact_path: str
    pip_requirements: str
    registered_model_name: str
    all_params: dict
    mlflow_uri: str

@dataclass(frozen=True)
class BiasDetectionConfig:
    results_path: Path
    embedding_model: str
    sentiment_model: str
    prompt_sentiment: str
    prompt_prob_sentiment: str