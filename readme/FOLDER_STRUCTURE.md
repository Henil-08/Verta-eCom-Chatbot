# **Verta-Chatbot Project Folder Structure**

## **Detailed Directory Structure**

```
root/
├── .github/
│   └── workflows/                # CI/CD workflows
│       ├── production.yml        # Workflow for production deployment
│       ├── staging.yml           # Workflow for staging deployment
├── Data_Pipeline/                # Scripts for data preparation
├── artifact/                     # Stores hash map for testing parquet files
├── config/                       # Configuration files
│   ├── config.yaml               # Workflow configurations
│   ├── prompts.yaml              # Prompt templates for the chatbot
├── evaluation/                   # Model evaluation scripts and results
│   ├── metrics/                  # Evaluation metrics storage
│   ├── results/                  # Evaluation results as parquets
│   ├── testset/                  # Test datasets for evaluation
├── logs/                         # Logs for debugging
├── media/                        # Various images for the readme folder
├── notebooks/                    # Notebooks containing the flow for better debugging
├── readme/                       # Readmes for the project
├── src/                          # Source code for the project
    ├── components/               # State, agents, and nodes for the langgraph architecture
    ├── config/                   # files for reading config.yaml and prompts.yaml file
    ├── constants/                # Constants
    ├── entity/                   # Entities for Pipelines
    ├── main/                     # Main codes for the pipelines
    ├── pipeline/                 # Workflow pipeline scripts
    ├── pydantic_models/          # Models used for the APIs
    ├── utils/                    # Various utility functions
    ├── app.py                    # Streamlit Website for the Chatbot
    ├── logger.py                 # Logger
    ├── main.py                   # Module for executing the pipeline flows
    ├── serve.py                  # FastAPI Wrapped Verta-chatbot
├── Dockerfile                    # Docker configuration for containerization
├── pyproject.toml                # Dependencies for the project managed by poetry
├── requirements.txt              # Dependency requirements for the project
└── README.md                     # Project documentation
```

---

## **Detailed File Descriptions**

### **1. .github/workflows/**
| **File Name**      | **Description**                                                                                      |
|--------------------|------------------------------------------------------------------------------------------------------|
| `staging.yml`      | Defines the staging CI/CD workflow. Includes unit testing, containerization, and deployment to staging. |
| `production.yml`   | Defines the production CI/CD workflow. Ensures staging validation before deploying to production.      |

### **2. config/**
| **File Name**      | **Description**                                                                                      |
|--------------------|------------------------------------------------------------------------------------------------------|
| `config.yaml`      | Core configurations for workflows, including environment variables and execution settings.            |
| `prompts.yaml`     | Stores predefined prompt templates for the chatbot, used during model training and inference.         |

### **3. evaluation/**
| **Subdirectory**   | **Description**                                                                                      |
|--------------------|------------------------------------------------------------------------------------------------------|
| `metrics/`         | Contains evaluation metrics such as precision, recall, and accuracy scores.                          |
| `results/`         | Stores detailed evaluation results in structured formats (e.g., JSON, Parquet).                      |
| `testset/`         | Includes test datasets in Parquet format for evaluating model performance.     

### **4. artifact/**
| **Purpose**        | **Description**                                                                                      |
|--------------------|------------------------------------------------------------------------------------------------------|
| Build Artifacts    | Stores build outputs such as Docker images and compiled artifacts.  |

### **5. notebooks/**
| **File Name**                       | **Description**                                                                                     |
|-------------------------------------|-----------------------------------------------------------------------------------------------------|
| `01_play_langchain_RAG.ipynb`       | Explores retrieval-augmented generation (RAG) using LangChain, showcasing basic use cases and setups.|
| `02_play_langgraph_RAG.ipynb`       | Experiments with RAG implementation using LangGraph and compares performance with LangChain.        |
| `03_improve_langgraph_chatbot.ipynb`| Focuses on optimizing LangGraph-based chatbot interactions for better query understanding and responses. |
| `04_play_postgres_integration.ipynb`| Demonstrates integration of PostgreSQL with RAG pipelines to manage and retrieve structured data.   |
| `05_generate_synthetic_questions.ipynb` | Generates synthetic data for training and evaluation, focusing on question-answer pairs.            |
| `06_play_rag_evaluation.ipynb`      | Evaluates the performance of the RAG pipeline using various metrics and datasets.                   |
| `trials.ipynb`                      | A sandbox notebook used for testing different configurations, code snippets, and experiments.       |

### **6. src/**
| **File Name**                       | **Description**                                                                                     |
|-------------------------------------|-----------------------------------------------------------------------------------------------------|
| `app.py`                            | Streamlit website for Verta.                                               |
| `serve.py`                          | Implements API endpoints for handling agent responses and citations using FastAPI.                  |
| `main.py`                          | Executes the pipelines in order.                  |

#### **Components**
| **File Name**                       | **Description**                                                                                     |
|-------------------------------------|-----------------------------------------------------------------------------------------------------|
| `agents.py`                         | Defines chatbot agents and their logic for handling user interactions.                              |
| `nodes.py`                          | Manages graph nodes used in LangGraph implementations.                                              |
| `sentiments.py`                     | Implements sentiment analysis modules for enriching user interaction data.                          |
| `state.py`                          | Manages application and user session states.                                                        |

#### **Config**
| **File Name**                       | **Description**                                                                                     |
|-------------------------------------|-----------------------------------------------------------------------------------------------------|
| `configuration.py`                  | Contains global configuration settings for the application.                                         |

#### **Entity**
| **File Name**                       | **Description**                                                                                     |
|-------------------------------------|-----------------------------------------------------------------------------------------------------|
| `config_entity.py`                  | Defines configuration schemas for different entities.                                               |

#### **Main**
| **File Name**                       | **Description**                                                                                     |
|-------------------------------------|-----------------------------------------------------------------------------------------------------|
| `bias_detection.py`                 | Implements bias detection logic for models.                                                        |
| `failure_detection.py`              | Monitors and identifies failures during pipeline execution.                                         |
| `graph.py`                          | Core graph logic for managing relationships between entities in LangGraph.                         |
| `model_evaluation.py`               | Evaluates models and outputs relevant metrics.                                                     |
| `test_ingestion.py`                 | Handles ingestion of test data for pipeline evaluation.                                             |

#### **Pipeline**
| **File Name**                       | **Description**                                                                                     |
|-------------------------------------|-----------------------------------------------------------------------------------------------------|
| `generation.py`                     | Contains code for handling the output when user hits APIs.                             |
| `stage_01_prepare_base_model.py`    | Prepares the base LangGraph model pipeline for training.                                            |
| `stage_02_test_data_ingestion.py`   | Processes and ingests test data required for model evaluation.                                      |
| `stage_03_model_evaluation.py`      | Runs model evaluation scripts and calculates metrics.                                               |
| `stage_04_bias_detection.py`        | Detects biases in model predictions and generates reports.                                          |
| `stage_05_failure_detection.py`     | Monitors and detects pipeline or model failures during execution.                                   |

#### **Utils**
| **File Name**                       | **Description**                                                                                     |
|-------------------------------------|-----------------------------------------------------------------------------------------------------|
| `common.py`                         | Contains common utility functions shared across modules.                                            |
| `database.py`                       | Manages database connections and operations.                                                       |

### **7. tests/**
| **File Name**                       | **Description**                                                                                     |
|-------------------------------------|-----------------------------------------------------------------------------------------------------|
| `test.py`                           | Contains test cases for validating various components of the application.                           |

### **8. readme/**
| **File Name**                       | **Description**                                                                                     |
|-------------------------------------|-----------------------------------------------------------------------------------------------------|
| `00_ML_PIPELINES.md`                | Documents the machine learning pipeline stages and their implementation.                            |
| `01_BASE_MODEL.md`                  | Details the structure, configuration, and setup of the base model.                                  |
| `02_TEST_INGESTION.md`              | Describes the process for ingesting and preparing test datasets.                                    |
| `03_MODEL_EVALUATION.md`            | Explains model evaluation metrics, methodologies, and expected outcomes.                           |
| `04_BIAS_DETECTION.md`              | Outlines the bias detection framework and its significance in pipeline evaluation.                  |
| `API_README.md`                     | Provides detailed instructions for API usage and endpoint definitions.                              |
| `CICD_WORKFLOW.MD`                  | Explains the CI/CD pipeline setup and its deployment workflow.                                      |
| `COST_ANALYSIS.md`                  | Analyzes resource costs associated with various parts of the project.                               |
| `FOLDER_STRUCTURE.md`               | Describes the overall folder structure and its organization.                                        |
| `LOGGING_MONITORING.MD`             | Documents the logging and monitoring systems implemented in the project.                           |
| `VERSION_ROLLBACK.md`               | Provides steps to rollback to a previous version of the system in case of failures.                 |

---

## Notes
- The `artifacts` folder contains generated artifacts such as hashmap files needed for model evaluation.
- `dvc.lock, dvc.yaml` files are necessary to run dvc pipelines
