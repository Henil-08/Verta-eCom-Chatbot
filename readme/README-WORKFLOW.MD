# **eCom-Chatbot CI/CD Pipeline and Deployment Workflow**

## **Overview**

The **eCom-Chatbot** project implements a robust Continuous Integration and Continuous Deployment (CI/CD) system to streamline development, testing, and production deployment processes. This pipeline includes:
- Automated unit and integration testing.
- Seamless containerization and storage in **Google Artifact Registry**.
- Scalable deployment to **Google Cloud Run**.
- Integration with LangGraph workflows and external APIs (e.g., OpenAI, HuggingFace).

This document provides a detailed explanation of the project structure, CI/CD workflows, testing integration, and deployment to staging and production environments.

---

## **Table of Contents**
1. [Project Features](#project-features)
2. [Directory Structure](#directory-structure)
3. [Detailed File Descriptions](#detailed-file-descriptions)
4. [Environment Variables](#environment-variables)
5. [CI/CD Workflow](#ci/cd-workflow)
    - [Staging Workflow](#staging-workflow)
    - [Production Workflow](#production-workflow)
6. [Artifact Registry and Cloud Run Integration](#artifact-registry-and-cloud-run-integration)
7. [Testing Framework](#testing-framework)
8. [Future Enhancements](#future-enhancements)

---

## **Project Features**

1. **Integrated Workflows**:
   - LangGraph workflows for pipeline orchestration.
   - Automated execution of stages like data ingestion, evaluation, and bias detection.

2. **Cloud-Native Deployment**:
   - Containerized deployments via Google Cloud Run.
   - Centralized image storage using Google Artifact Registry.

3. **CI/CD Pipelines**:
   - Staging and production pipelines with GitHub Actions.
   - Integrated testing for application reliability.

4. **Comprehensive Testing**:
   - Unit tests for API endpoints, Python workflows, and LangGraph nodes.
   - Automated test result reporting.

---

## **Directory Structure**

```
root/
├── .github/
│   └── workflows/                # CI/CD workflows
│       ├── production.yml        # Workflow for production deployment
│       ├── staging.yml           # Workflow for staging deployment
├── Data_Pipeline/                # Scripts for data preparation
├── artifact/                     # Artifacts and build outputs
├── config/                       # Configuration files
│   ├── config.yaml               # Workflow configurations
│   ├── prompts.yaml              # Prompt templates for the chatbot
├── evaluation/                   # Model evaluation scripts and results
│   ├── metrics/                  # Evaluation metrics storage
│   ├── results/                  # Evaluation results
│   ├── testset/                  # Test datasets for evaluation
├── logs/                         # Logs for debugging
├── pipeline/                     # Workflow pipeline scripts
│   ├── stage_01_prepare_base_model.py
│   ├── stage_02_test_data_ingestion.py
│   ├── stage_03_model_evaluation.py
│   ├── stage_04_bias_detection.py
│   ├── stage_05_failure_detection.py
├── Dockerfile                    # Docker configuration for containerization
└── README.md                     # Project documentation
```

---

## **Detailed File Descriptions**

### **1. .github/workflows/**
| **File Name**      | **Description**                                                                                      |
|--------------------|------------------------------------------------------------------------------------------------------|
| `staging.yml`      | Defines the staging CI/CD workflow. Includes unit testing, containerization, and deployment to staging. |
| `production.yml`   | Defines the production CI/CD workflow. Ensures staging validation before deploying to production.      |

### **2. pipeline/**
| **File Name**                       | **Description**                                                                                      |
|-------------------------------------|------------------------------------------------------------------------------------------------------|
| `stage_01_prepare_base_model.py`    | Prepares the base LangGraph model pipeline for training.                                             |
| `stage_02_test_data_ingestion.py`   | Processes and ingests test data required for model evaluation.                                       |
| `stage_03_model_evaluation.py`      | Runs model evaluation scripts and calculates metrics.                                                |
| `stage_04_bias_detection.py`        | Detects biases in model predictions and generates reports.                                           |
| `stage_05_failure_detection.py`     | Monitors and detects pipeline or model failures during execution.                                    |

### **3. config/**
| **File Name**      | **Description**                                                                                      |
|--------------------|------------------------------------------------------------------------------------------------------|
| `config.yaml`      | Core configurations for workflows, including environment variables and execution settings.            |
| `prompts.yaml`     | Stores predefined prompt templates for the chatbot, used during model training and inference.         |

### **4. evaluation/**
| **Subdirectory**   | **Description**                                                                                      |
|--------------------|------------------------------------------------------------------------------------------------------|
| `metrics/`         | Contains evaluation metrics such as precision, recall, and accuracy scores.                          |
| `results/`         | Stores detailed evaluation results in structured formats (e.g., JSON, Parquet).                      |
| `testset/`         | Includes test datasets in Parquet format for evaluating model performance.                           |

### **5. artifact/**
| **Purpose**        | **Description**                                                                                      |
|--------------------|------------------------------------------------------------------------------------------------------|
| Build Artifacts    | Stores build outputs such as Docker images and compiled artifacts.                                   |

---

## **Environment Variables**

Environment variables are critical for securing sensitive data and managing configurations.

| **Variable**                  | **Description**                                |
|--------------------------------|-----------------------------------------------|
| `OPENAI_API_KEY`              | API key for OpenAI integration.               |
| `HF_TOKEN`                    | HuggingFace token for embeddings.             |
| `LANGFUSE_PUBLIC_KEY`         | Public key for LangFuse analytics.            |
| `LANGFUSE_SECRET_KEY`         | Secret key for LangFuse analytics.            |
| `DB_USER`                     | PostgreSQL database username.                 |
| `DB_PASS`                     | PostgreSQL database password.                 |
| `DB_NAME`                     | PostgreSQL database name.                     |
| `INSTANCE_CONNECTION_NAME`    | Connection name for Cloud SQL instance.       |
| `CLOUD_RUN_SERVICE`           | Name of the Cloud Run service.                |
| `PROJECT_ID`                  | Google Cloud project ID.                      |

---

## **CI/CD Workflow**

The CI/CD pipeline automates the entire lifecycle from testing to deployment.

### **1. Staging Workflow**
#### **File:** `staging.yml`
- **Trigger:** Push to the `staging` branch.
- **Purpose:** Deploys to the staging environment for validation and testing.

**Steps:**
1. **Checkout Code:**
   - Pulls the latest changes from the `staging` branch.
2. **Run Unit Tests:**
   - Executes unit tests for APIs and pipelines using `pytest`.
3. **Build Docker Image:**
   - Creates a Docker container for the backend application.
4. **Push to Artifact Registry:**
   - Pushes the containerized application to Google Artifact Registry.
5. **Deploy to Cloud Run (Staging):**
   - Deploys the container to a staging Cloud Run service for testing.

---

### **2. Production Workflow**
#### **File:** `production.yml`
- **Trigger:** Push to the `production` branch.
- **Purpose:** Deploys validated code to the production environment.

**Steps:**
1. **Sync Validation:**
   - Ensures the `production` branch is in sync with the validated `staging` branch.
2. **Authentication:**
   - Authenticates with Google Cloud using a service account key.
3. **Deploy to Cloud Run (Production):**
   - Deploys the containerized application to the production Cloud Run service.
4. **Post-Deployment Validation:**
   - Ensures the production service is live and accessible.

---

## **Artifact Registry and Cloud Run Integration**

### **Artifact Registry**
1. **Build Docker Image:**
   - The application is containerized using the `Dockerfile`:
   ```bash
   docker build -t "us-east1-docker.pkg.dev/{PROJECT_ID}/{GAR_NAME}/{SERVICE}:{COMMIT_SHA}" -f Dockerfile .
   ```
2. **Push to Registry:**
   - Pushes the Docker image to Google Artifact Registry:
   ```bash
   docker push "us-east1-docker.pkg.dev/{PROJECT_ID}/{GAR_NAME}/{SERVICE}:{COMMIT_SHA}"
   ```

### **Cloud Run Deployment**
1. **Configuration:**
   - Cloud Run is configured with the following:
     - **Memory:** 16Gi
     - **CPU:** 4
     - **Autoscaling:** Min 1, Max 10 instances
   - Example Deployment Command:
   ```bash
   gcloud run deploy {SERVICE_NAME} \
       --image "us-east1-docker.pkg.dev/{PROJECT_ID}/{GAR_NAME}/{SERVICE}:{COMMIT_SHA}" \
       --region "us-east1" \
       --platform managed \
       --allow-unauthenticated \
       --memory "16Gi" \
       --cpu "4"
   ```

---

## **Testing Framework**

1. **Backend Tests:**
   - Tests all API endpoints and pipeline stages.
   - Located in `tests/` directory.
   - Example Command:
     ```bash
     pytest tests/ --junitxml=test-results.xml
     ```

2. **Test Reporting:**
   - Test results are uploaded as artifacts for CI/CD pipelines.
