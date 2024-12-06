# **eCom-Chatbot: Model Rollback, Evaluation, and Deployment Pipelines**

## **Overview**

The **eCom-Chatbot** is a scalable, cloud-native solution integrating robust pipelines for model evaluation and deployment. The system employs **Google Cloud Run** for serverless application hosting and **Google Artifact Registry** for containerized artifact storage. A core feature of the system is the **rollback process** in Cloud Run, which ensures stability by allowing quick reversion to previous stable deployments in case of errors. Additionally, evaluation pipelines validate model performance, while failure detection ensures reliable operation.

This document details the rollback mechanism, model evaluation pipeline, failure detection, and deployment workflows.

---

## **Table of Contents**

1. [Cloud Run Rollback Process](#cloud-run-rollback-process)
    - [How Rollback Works in Cloud Run](#how-rollback-works-in-cloud-run)
    - [Detailed Steps for Rollback](#detailed-steps-for-rollback)
    - [Best Practices for Rollback](#best-practices-for-rollback)
2. [Environment Variables](#environment-variables)
3. [Pipeline Overview](#pipeline-overview)
    - [Model Evaluation Pipeline](#model-evaluation-pipeline)
    - [Failure Detection Pipeline](#failure-detection-pipeline)
4. [Deployment Workflow](#deployment-workflow)
    - [Artifact Registry Integration](#artifact-registry-integration)
    - [Cloud Run Deployment](#cloud-run-deployment)
5. [Future Enhancements](#future-enhancements)

---

## **Cloud Run Rollback Process**
![Google Cloud System Rollback](/media/image_rollback.png)

### **How Rollback Works in Cloud Run**

In **Google Cloud Run**, every deployment creates a new **revision**, which is a snapshot of the service at that point. If a new deployment introduces bugs or errors, rollback allows redirecting all traffic to a previously stable revision. This ensures service continuity with minimal downtime.

Key features of the rollback process:
- **Revision History**: Each deployment revision is retained, with unique identifiers and metadata.
- **Traffic Management**: Traffic can be redirected to any previous revision.
- **No Downtime Rollback**: Cloud Run enables instant rollback without needing to rebuild or redeploy.

### **Detailed Steps for Rollback**

1. **Access Cloud Run Console**:
   - Open **Google Cloud Console**.
   - Navigate to **Cloud Run** and select your service (e.g., `verta-chat-service`).

2. **View Revisions**:
   - Click on the **"Revisions"** tab to see the history of deployments.
   - Each revision is listed with details such as timestamp, deployment ID, and current traffic percentage.

3. **Select a Stable Revision**:
   - Identify the last stable revision (e.g., `verta-chat-service-00012`).

4. **Manage Traffic**:
   - Click **"Manage Traffic"**.
   - Set the desired revision to handle **100% of traffic**.
   - Save the changes.

5. **Validation**:
   - Confirm rollback by testing the application endpoints.
   - Monitor logs in the Cloud Run dashboard to ensure proper functionality.

6. **Monitor and Adjust**:
   - Use the **Logs** and **Metrics** tabs in Cloud Run to monitor post-rollback performance.

### **Best Practices for Rollback**

1. **Use Descriptive Revision Tags**:
   - Tag revisions with meaningful names for easier identification.

2. **Test Before Rollback**:
   - Use the **split traffic** feature to test a revision with a small percentage of traffic before a full rollback.

3. **Automate Rollbacks**:
   - Integrate rollback triggers in your CI/CD pipeline for automated recovery during failure detection.

---

## **Environment Variables**

Environment variables manage sensitive information and configuration across pipelines.

| **Variable**                  | **Description**                                |
|--------------------------------|-----------------------------------------------|
| `HF_TOKEN`                    | HuggingFace token for embeddings.             |
| `OPENAI_API_KEY`              | API key for OpenAI integration.               |
| `GROQ_API_KEY`                | API key for Groq service.                     |
| `GOOGLE_APPLICATION_CREDENTIALS` | Path to GCP service account JSON key.         |
| `LANGFUSE_PUBLIC_KEY`         | Public key for LangFuse analytics.            |
| `LANGFUSE_SECRET_KEY`         | Secret key for LangFuse analytics.            |
| `MLFLOW_TRACKING_URI`         | MLFlow tracking server URI.                   |
| `INSTANCE_CONNECTION_NAME`    | GCP connection name for Cloud SQL instance.   |
| `DB_USER`                     | PostgreSQL database username.                 |
| `DB_PASS`                     | PostgreSQL database password.                 |

---

## **Pipeline Overview**

### **Model Evaluation Pipeline**

#### **Purpose**
Evaluates deployed models for performance metrics such as accuracy, recall, and bias detection. Logs results to **MLFlow** for tracking.

#### **Steps**
1. **Initialize Workflow**:
   - Use `PrepareBaseTrainingPipeline` to initialize LangGraph workflows.

2. **Configure Evaluation**:
   - Load evaluation configurations from `config.yaml`.

3. **Run Evaluation**:
   - Execute evaluation using `Evaluation` class.
   - Save metrics and results.

4. **Log to MLFlow**:
   - Track evaluation metrics and parameters in MLFlow.

---

### **Failure Detection Pipeline**

#### **Purpose**
Monitors logs and model outputs for anomalies, triggering alerts for mitigation.

#### **Steps**
1. **Initialize Detection**:
   - Load configurations for failure monitoring.

2. **Detect Failures**:
   - Run `FailureDetection` to identify issues in logs and predictions.

3. **Send Alerts**:
   - Notify stakeholders via **MS Teams Webhook** or similar tools.

---

## **Deployment Workflow**

### **Artifact Registry Integration**

#### **Purpose**
Manages containerized application images.

#### **Steps**
1. **Build Docker Image**:
   ```bash
   docker build -t "us-east1-docker.pkg.dev/{PROJECT_ID}/{GAR_NAME}/{SERVICE}:{COMMIT_SHA}" -f Dockerfile .
   ```

2. **Push Image to Artifact Registry**:
   ```bash
   docker push "us-east1-docker.pkg.dev/{PROJECT_ID}/{GAR_NAME}/{SERVICE}:{COMMIT_SHA}"
   ```

3. **Retrieve Image for Deployment**:
   - Deployment pipelines pull images directly from Artifact Registry.

---

### **Cloud Run Deployment**

#### **Steps**
1. **Deploy Container**:
   ```bash
   gcloud run deploy {SERVICE_NAME} \
       --image "us-east1-docker.pkg.dev/{PROJECT_ID}/{GAR_NAME}/{SERVICE}:{COMMIT_SHA}" \
       --region "us-east1" \
       --platform managed \
       --allow-unauthenticated
   ```

2. **Post-Deployment Validation**:
   - Test endpoints and monitor metrics for performance.
