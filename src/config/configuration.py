from pathlib import Path

from constants import CONFIG_FILE_PATH, PROMPTS_FILE_PATH
from utils.common import read_yaml, create_directories
from entity.config_entity import (EvaluationConfig, PrepareBaseModelConfig, TestIngestionConfig, BiasDetectionConfig)


class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        prompts_filepath = PROMPTS_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.prompts = read_yaml(prompts_filepath)


    def get_prepare_base_model_config(self) -> PrepareBaseModelConfig:
        config = self.config.prepare_base_model

        create_directories([config.cache_dir, config.faiss_dir, config.meta_dir])
        
        prepare_base_model_config = PrepareBaseModelConfig(
            cache_dir=config.cache_dir,
            faiss_dir=config.faiss_dir,
            meta_dir=config.meta_dir,
            supervisor_model=config.supervisor_model,
            metadata_model=config.metadata_model,
            base_model=config.base_model,
            followup_model=config.followup_model,
            prompt_supervisor=self.prompts.SUPERVISOR_PROMPT,
            prompt_metadata=self.prompts.METADATA_PROMPT,
            prompt_base_model=self.prompts.BASE_MODEL_PROMPT,
            prompt_followup=self.prompts.FOLLOWUP_PROMPT,
        )

        return prepare_base_model_config
    

    def get_test_ingestion_config(self) -> TestIngestionConfig:
        config = self.config.test_ingestion

        create_directories([config.artifact_dir, config.evaluation_root_dir, config.testset_path])

        test_ingestion_config = TestIngestionConfig(
            artifact_dir=Path(config.artifact_dir),
            evaluation_root_dir=Path(config.evaluation_root_dir),
            testset_path=Path(config.testset_path),
            prompt_metadata=self.prompts.METADATA_PROMPT,
        )
        return test_ingestion_config


    def get_evaluation_config(self) -> EvaluationConfig:
        config = self.config.evaluation
        base_config = self.config.prepare_base_model

        supervisor = '-'.join(base_config.supervisor_model.split('-')[:-1])
        metadata = '-'.join(base_config.metadata_model.split('-')[:-1])
        base_model = '-'.join(base_config.base_model.split('-')[:-1])
        followup_model = '-'.join(base_config.followup_model.split('-')[:-1])

        create_directories([config.root_dir])
        create_directories([config.metrics_path])
        create_directories([config.results_path])
        create_directories([config.testset_path])

        eval_config = EvaluationConfig(
            root_dir=Path(config.root_dir),
            metrics_path=Path(config.metrics_path),
            results_path=Path(config.results_path),
            testset_path=Path(config.testset_path),
            file_hash=Path(config.file_hash),
            lc_model="src/graph.py",
            artifact_path="langgraph",
            pip_requirements="requirements.txt",
            registered_model_name=f"verta-{supervisor}-{metadata}-{base_model}-{followup_model}",
            all_params=self.prompts,
            mlflow_uri="https://dagshub.com/eCom-dev5/eCom-Chatbot.mlflow",
        )
        return eval_config
    

    def get_bias_detection_config(self) -> BiasDetectionConfig:
        config = self.config.bias_detection

        bias_detection_config = BiasDetectionConfig(
            results_path=config.results_path,
            embedding_model=config.embedding_model,
            sentiment_model=config.sentiment_model,
            prompt_sentiment=self.prompts.SENTIMENT_PROMPT,
            prompt_prob_sentiment=self.prompts.SENTIMENT_PROBABILITY_PROMPT,
        )
        return bias_detection_config