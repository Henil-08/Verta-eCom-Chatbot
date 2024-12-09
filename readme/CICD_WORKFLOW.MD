# **Verta-Chatbot CI/CD Pipeline and Deployment Workflow**

## **Overview**

The **Verta-Chatbot** project implements a robust Continuous Integration and Continuous Deployment (CI/CD) system to streamline development, testing, and production deployment processes. This pipeline includes:
- Automated unit and integration testing.
- Seamless containerization and storage in **Google Artifact Registry**.
- Scalable deployment to **Google Cloud Run**.
- Integration with LangGraph workflows and external APIs (e.g., OpenAI, HuggingFace).

This document provides a detailed explanation of the project structure, CI/CD workflows, testing integration, and deployment to staging and production environments.

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

## **CI/CD Workflow**

The CI/CD pipeline automates the entire lifecycle from testing to deployment.

### Integration with GitHub Workflows
The integration of these workflows within the CI/CD pipeline occurs as follows:

- **Staging Deployment**: The `staging.yml` workflow runs unit tests, evaluates the model including the bias detection, to the staging environment. This stage is crucial for real-world testing under controlled conditions.

- **Production Deployment**: The `production.yml` workflow handles the deployment of the models to the production environment after ensuring that all criteria are met in the staging review. This workflow also triggers the final model training jobs in production, processing new data as needed.

### Development Phase
   - **Activity**: Developers push code to branches od the dev branch.
   - **Details**: This phase may include local tests and initial validations before moving the code to a shared environment.

### Pull Request Creation
- **Activity**: Once development is completed in the `dev` branch, a pull request is created to merge the changes into the `staging` branch.
- **Details**: This triggers several checks and workflows.

### Staging Workflow
#### **File:** `staging.yml`
- **Trigger:** Pull request from `dev` to the `staging` branch.
- **Purpose:** Deploys to the staging environment for validation and testing.

**Steps:**
1. **Checkout Code:**
   - Pulls the latest changes from the `staging` branch.
2. **Run Model Pipeline:**
   - Execute `main.py` for model pipelines. Sends alerts and notifications if the metrics are not upto the mark.
3. **Run Unit Tests:**
   - Executes unit tests for APIs and pipelines using `pytest`.
4. **Build Docker Image:**
   - Creates a Docker container for the backend application.
5. **Push to Artifact Registry:**
   - Pushes the containerized application to Google Artifact Registry.
6. **Deploy to Cloud Run (Staging):**
   - Deploys the container to a staging Cloud Run service for testing.

![Staging Workflow](../media/Staging.png)
![Staging](../media/Staging_1.png)

### Pull Request Creation
- **Activity**: Once development is completed in the `staging` branch, a pull request is created to merge the changes into the `production` branch.
- **Details**: This triggers several checks and workflows.


### Production Workflow
#### **File:** `production.yml`
- **Trigger:** Pull request from `staging` to the `production` branch.
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

![Production](../media/production.png)
![Production](../media/production_1.png)

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