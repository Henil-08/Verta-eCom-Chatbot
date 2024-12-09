# eCommerce-Chatbot using LangGraph (Verta)


## Introduction

The **Verta Chatbot** project is a sophisticated AI-driven solution designed to enhance user interactions with product information. Deployed as a serverless **FASTAPI** API on **Cloud Run**, it serves users by answering questions about products, leveraging both metadata and user reviews for context. The chatbot utilizes a multi-agent workflow, with distinct agents performing specialized roles. A **Metadata Agent** is responsible for summarizing product descriptions, while a **Retriever Agent** fetches relevant data from a vector store containing user reviews. This allows the chatbot to answer a wide range of user inquiries, drawing on both product details and feedback from other customers.

The database for this solution is hosted on **PostgreSQL** in **Google Cloud Platform (GCP)**, ensuring scalable, reliable storage. The project utilizes **CI/CD pipelines** via **GitHub Actions**, automating code deployment and ensuring seamless integration and delivery. Additionally, the system incorporates **LLM-as-Judge** to generate synthetic test questions for a set of products, while **bias detection** algorithms analyze potential biases in the chatbot's responses to ensure fairness and accuracy.

Experiment tracking is managed through **MLflow**, which captures model performance and experiment metadata, while **Langfuse** is used for tracing user interactions and gathering feedback to continuously improve the system. For monitoring and alerting, **GCP Logs** are utilized with notifications configured to send alerts to a **Teams channel** for real-time system health checks.

The chatbot’s data orchestration is powered by **Apache Airflow**, while **FaisDB** is used as the vector store for storing product reviews and context. The system integrates three LLMs — **GPT-4o-Mini**, **Llama 3.1-70B**, and **Llama 3.1-8B** — running on four nodes to support its operations. The chatbot’s multi-agent flow is managed using **LangGraph**, a framework for orchestrating complex workflows. For ease of use, the chatbot is also available as a **Streamlit web app**, with integration capabilities for custom frontends via API, secured using **Auth Bearer Tokens**.

## Project Architecture
![Verta Achitecture](media/Verta%20Architecture.png)

## Getting Started - Guide
To start working with the `ecom-chatbot` project, please follow the setup instructions outlined in the [Project Setup Guide](https://github.com/eCom-dev5/eCom-Chatbot/blob/dev/readme/SETUP.md).

This guide includes steps for creating a GCP account, configuring environment variables, setting up GitHub secrets, deploying locally, and hosting the Chatbot API. Once the initial setup is complete, you can explore the following detailed guides for specific aspects of the project:

- [Understand the API Payloads](https://github.com/eCom-dev5/eCom-Chatbot/blob/dev/readme/API_README.md)
- [Explore ML Pipelines](https://github.com/eCom-dev5/eCom-Chatbot/blob/dev/readme/00_ML_PIPELINES.md)
- [Stage 1 - Base Model](https://github.com/eCom-dev5/eCom-Chatbot/blob/dev/readme/01_BASE_MODEL.md)
- [Stage 2 - Test Data Ingestion](https://github.com/eCom-dev5/eCom-Chatbot/blob/dev/readme/02_TEST_INGESTION.md)
- [Stage 3 - Model Evaluation](https://github.com/eCom-dev5/eCom-Chatbot/blob/dev/readme/03_MODEL_EVALUATION.md)
- [Stage 4 - Bias Detection](https://github.com/eCom-dev5/eCom-Chatbot/blob/dev/readme/04_BIAS_DETECTION.md)
- [CI/CD Workflow](https://github.com/eCom-dev5/eCom-Chatbot/blob/dev/readme/CICD_WORKFLOW.MD)
- [Cost Analysis](https://github.com/eCom-dev5/eCom-Chatbot/blob/dev/readme/COST_ANALYSIS.md)
- [Logging and Monitoring Setup](https://github.com/eCom-dev5/eCom-Chatbot/blob/dev/readme/LOGGING_MONITORING.MD)
- [Version Rollback](https://github.com/eCom-dev5/eCom-Chatbot/blob/dev/readme/VERSION_ROLLBACK.md)
- [Understand the Project Folder Structure](https://github.com/eCom-dev5/eCom-Chatbot/blob/dev/readme/FOLDER_STRUCTURE.md)