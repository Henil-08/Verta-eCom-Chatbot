# **Cost Analysis and Resource Usage for Hosting Verta Chatbot on GCP**

This report provides a detailed breakdown of the costs incurred while deploying and operating the **Verta Chatbot** on **Google Cloud Platform (GCP)**. We'll go over each service used, what drove the costs, and how these costs can be optimized moving forward. The goal is to ensure efficient usage of resources while maintaining high performance and scalability for the project.

---

## **Overview**

The total cost for this billing cycle amounted to **$208.60** for the month of November 2024, with the following key contributors:
1. **Cloud SQL**: The largest expense due to heavy database usage for storing and querying metadata, logs, and evaluation datasets.
2. **Cloud Run**: Costs associated with serverless hosting for APIs and workflows.
3. **Container Registry Vulnerability Scanning**: Reflecting our commitment to secure and compliant deployments.

The structure and the need for these services are directly tied to the project’s complexity and scalability requirements.

---
![Google Cloud Cost ](/media/costreport.png)

## **Cost Breakdown**

| **Service**                                  | **Cost (USD)** | **What It Does**                                                                                 |
|----------------------------------------------|----------------|--------------------------------------------------------------------------------------------------|
| **Cloud SQL**                                | $110.31        | Central database for storing metadata, evaluation data, and user queries.                        |
| **Cloud Run**                                | $63.77         | Hosts the serverless backend, handling LangGraph workflows and user interactions.                |
| **Container Registry Vulnerability Scanning**| $26.78         | Scans container images in the Artifact Registry for vulnerabilities before deployment.           |
| **Artifact Registry**                        | $6.79          | Stores Docker images for backend services and workflows.                                         |
| **Cloud Storage**                            | $0.55          | Temporary storage for logs, artifacts, and evaluation outputs.                                   |
| **Vertex AI**                                | $0.36          | Tracks evaluation metrics and experiments to compare model performance.                          |
| **Networking**                               | $0.04          | Internal and external traffic costs between Cloud Run, Cloud SQL, and other services.            |
| **Compute Engine**                           | $0.00          | Not actively used during this billing cycle.                                                     |

---
# **ChatGPT Model Token Cost**

As part of the chatbot architecture, ChatGPT (GPT-4.0 Mini) is used for generating responses. Token usage and associated costs are monitored to ensure optimal resource utilization.

---

## **Token Costs**

| **Model Name**  | **Input Cost (/1k tokens)** | **Output Cost (/1k tokens)** | **Total Cost (/1k tokens)** |
|------------------|-----------------------------|------------------------------|-----------------------------|
| **GPT-4 Mini**   | $0.01                      | $0.03                       | $0.04                      |

---

## **Example Cost Calculation**

For a query with **1,500 input tokens** and **450 output tokens**:

- **Input Cost**: `1.5 × 0.01 = $0.015`
- **Output Cost**: `0.45 × 0.03 = $0.0135`
- **Total Query Cost**: `$0.015 + $0.0135 = $0.0285`

---

## **Monthly Token Usage Breakdown**

| **Metric**                 | **Value**           |
|-----------------------------|---------------------|
| **Average Query Tokens**    | 2,500 (Input + Output) |
| **Monthly Queries**         | 50,000             |
| **Total Tokens**            | 125,000,000        |
| **Estimated Monthly Cost**  | $5,000             |


### **1. Cloud SQL ($110.31)**

- **Why the cost is high**: Cloud SQL is our backbone for data storage, handling a significant amount of read/write operations for:
  - Storing product metadata.
  - Maintaining chatbot interaction logs.
  - Running queries during evaluation.
- **Breakdown**:
  - Instance type: High I/O to handle real-time queries.
  - Backups: Enabled for disaster recovery.
  - Persistent connection between Cloud Run and Cloud SQL.

#### **Optimization Steps**:
1. Tune database queries to minimize redundant operations.
2. Schedule backups during off-peak hours to avoid additional load.
3. Downgrade instance type during low-traffic hours or non-peak times.

---

### **2. Cloud Run ($63.77)**

- **Why the cost is high**: Cloud Run is handling all backend services, including LangGraph workflows, APIs, and user interactions. The autoscaling setup means more instances spin up during testing and evaluation phases.
- **Breakdown**:
  - Autoscaling limits: 1 to 10 instances.
  - Handles API requests, LangGraph pipelines, and model evaluation.

#### **Optimization Steps**:
1. Reduce **minimum instances** to `0` to save costs during idle periods.
2. Shorten **timeout settings** to avoid resource locking for long requests.
3. Fine-tune concurrency to maximize the usage of each instance.

---

### **3. Container Registry Vulnerability Scanning ($26.78)**

- **Why the cost is high**: We perform regular vulnerability scans on all container images stored in the Artifact Registry to ensure security compliance.
- **What it’s doing**:
  - Scans Docker images for vulnerabilities before deployment.

#### **Optimization Steps**:
1. Schedule scans only for major updates or new images.
2. Exclude stable, unchanged images from frequent scans.

---

### **4. Artifact Registry ($6.79)**

- **Why it’s needed**: This is where all our Docker images live. Each build (from staging to production) is pushed here, ensuring proper version control and easy rollbacks.
- **Cost driver**: Storage of multiple versions of images.

#### **Optimization Steps**:
1. Enable lifecycle rules to automatically delete old or unused images.
2. Review and manually remove deprecated image versions.

---

### **5. Cloud Storage ($0.55)**

- **Why it’s needed**: Temporary storage for evaluation results, logs, and artifacts generated during workflows.
- **Cost driver**: Small-scale usage for logs and intermediary data.

#### **Optimization Steps**:
1. Move older logs to **Nearline Storage** for cheaper long-term retention.
2. Automate deletion of stale data after 30 days.

---

### **6. Vertex AI ($0.36)**

- **What it’s for**: Tracks experiment results and evaluation metrics, allowing us to compare different models.
- **Cost driver**: Light usage for metrics logging.

#### **Optimization Steps**:
1. Log only essential fields to reduce data storage overhead.
2. Archive older experiments after analysis.

---

### **7. Networking ($0.04)**

- **Why it’s needed**: Data transfer between services like Cloud Run and Cloud SQL.
- **Cost driver**: Minimal due to efficient design.

#### **Optimization Steps**:
1. Use private IP connections where possible to minimize external traffic costs.
2. Reduce unnecessary API calls between services.

---

## **Key Observations**

1. **Cloud SQL takes the majority of the cost** (52.87% of the total). This is expected given its critical role in the project.
2. **Cloud Run follows closely** (30.58%), driven by autoscaling during intensive operations like evaluation and testing.
3. **Security costs from Container Registry Scanning** (12.84%) reflect our focus on safe deployments.
4. **Artifact Registry and Cloud Storage** make up minor but essential costs, showing efficient resource usage.

---

## **Cost Summary**

- **Total Cost**: $208.60
  - Cloud SQL: 52.87%
  - Cloud Run: 30.58%
  - Container Scanning: 12.84%
  - Other services: 3.71%

---

## **Optimization Opportunities**

### **Short-Term**
1. Lower minimum instance counts in **Cloud Run** during off-hours.
2. Schedule **vulnerability scans** only for major updates.
3. Delete old Docker images from **Artifact Registry**.

### **Long-Term**
1. Use **Coldline Storage** for archiving older logs and evaluation data.
2. Consolidate metrics tracking to **Vertex AI** for reduced redundancy.
3. Automate instance scaling and downgrading for **Cloud SQL** during non-peak periods.

---

## **Structure of the Cost**

### **What caused the costs to rise?**
1. Heavy database usage from Cloud SQL due to real-time queries and storage needs.
2. Cloud Run’s autoscaling handled intense evaluation and testing workloads.
3. Regular container scans added a consistent layer of expense.

### **How are the costs justified?**
These costs reflect the project’s **scalable infrastructure**:
- Real-time analytics and processing in Cloud Run.
- Secure image deployments via Container Registry.
- Centralized data storage and querying through Cloud SQL.

---

## **Future Cost Monitoring**

To keep costs under control, we’ll:
1. **Set budget alerts** in GCP to monitor expenses.
2. **Enable cost dashboards** for real-time tracking of each service.
3. **Implement automation** for scaling, archiving, and cleanup tasks.
