from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PrepareBaseModelConfig:
    supervisor_model: str
    metadata_model: str
    base_model: str
    followup_model: str
    prompt_supervisor: str
    prompt_metadata: str
    prompt_base_model: str
    prompt_followup: str


@dataclass(frozen=True)
class EvaluationConfig:
    root_dir: Path
    metrics_path: Path
    results_path: Path
    testset_path: Path
    lc_model: str
    artifact_path: str
    pip_requirements: str
    registered_model_name: str
    all_params: dict
    mlflow_uri: str