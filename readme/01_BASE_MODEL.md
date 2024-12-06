# **'Verta Chatbot' Base Model RAG Chatbot Architecture**

## **Overview**

The **Base Model RAG (Retrieval Augment-Generation)** Chatbot architecture is a robust solution for processing user queries in real time with high accuracy and contextual relevance. By combining advanced retrieval techniques with state-of-the-art generative models, the chatbot provides precise and engaging responses. The system seamlessly integrates multiple components, including retrieval systems, metadata summarization modules, and language models, all orchestrated by a centralized **Supervisor Module**.

Additionally, the system features integration with **LangFuse Analytics**, which ensures comprehensive monitoring of query traces, token usage, and cost efficiency. This modular and scalable architecture is optimized for real-world applications in product information retrieval, customer support, and interactive recommendations.

---

## **Detailed Architecture**
![System Architecture](/media/workflow.jpeg)
### **Core Components**

1. **Supervisor Module**:  
  **Model:** gpt-4o-mini
   The central decision-making module that dynamically routes queries to appropriate processing pipelines.

2. **Metadata**:  
  **Model:** llama3.1-8b
   Summarizes retrieved structured metadata into concise, user-friendly responses.

3. **Vectorstore Retriever**:  
  **Database:** FAISS Vectorstore DB
   Handles the retrieval of unstructured data, including long-form user reviews and product descriptions, using FAISS-based similarity searches.

4. **Generate**:  
  **Model:** llama3.1-70b
   Combines contextual data, metadata summaries, and user inputs to generate the final response.

5. **Follow-up Question Generator**:  
  **Model:** llama3.1-8b
   Dynamically formulates follow-up questions to enhance user engagement and clarify incomplete queries.

6. **LangFuse Analytics**:  
   Monitors the entire pipeline, providing metrics on trace logs, token usage, API performance, and query costs.

---

### **Architecture Workflow**

```mermaid
flowchart TD
    start[User Query]
    start --> supervisor["Supervisor (Gpt-4o-mini)"]
    supervisor -->|Metadata Query| summarizer["Metadata Summarizer (Llama 3.1 8b)"]
    supervisor -->|Unstructured Query| vectorstore["Vectorstore Retriever"]
    vectorstore --> mainllm["Main LLM (LLaMA 3.1 70B)"]
    summarizer --> mainllm
    mainllm --> followup["Follow-up Question Generator (Llama 3.1 8b)"]
    followup --> response["Response to User"]
    response --> langfuse["LangFuse Analytics"]
```

---

## **Component Details**

### **1. Supervisor Module**
- **Role**:
   - Acts as the decision-making layer, routing queries based on their type (metadata vs. unstructured).
- **Workflow**:
   - Receives user queries through the `dev/stream` API.
   - Routes queries to either the Metadata Retriever or the Vectorstore Retriever.
- **Integration**:
   - Logs routing decisions and processing times via LangFuse Analytics for transparency and performance monitoring.

---

### **2. Metadata Summarizer**
- **Role**:
   - Summarizes preloaded structured data, such as product metadata, from APIs or vector databases.
- **Workflow**:
   - Outputs structured data for downstream processing by the Metadata Summarizer.
- **Use Case**:
   - Querying product-specific information like “What are the features of Product X?”
- **LangFuse Integration**:
   - Tracks API calls, retrieval efficiency, and associated costs.

---

### **3. Vectorstore Retriever**
- **Role**:
   - Retrieves unstructured textual data (e.g., reviews, descriptions) using vector embeddings.
- **Workflow**:
   - Converts textual data into vector embeddings using HuggingFace's MiniLM.
   - Stores and retrieves embeddings using FAISS for fast similarity-based searches.
   - Fetches contextually relevant documents for user queries.
- **Example**:
   - Query: "What do users say about noise cancellation in this product?"
   - Retrieves: Top relevant reviews mentioning noise cancellation.
- **LangFuse Integration**:
   - Tracks token usage for embeddings and retrieval operations.

---

### **4. Metadata Summarizer**
- **Role**:
   - Summarizes structured data into concise and readable formats.
- **Workflow**:
   - Processes structured metadata (e.g., product specs, features, pricing).
   - Outputs a human-readable summary for the Main LLM to use in response generation.
- **Example**:
   - Input: Product metadata.
   - Output: "This product features lightweight design, noise cancellation, and a 10-hour battery life."
- **LangFuse Integration**:
   - Monitors token usage during summarization.

---

### **5. Main LLM (e.g., ChatGPT or LLaMA)**
- **Role**:
   - Synthesizes a comprehensive response by combining:
     - User inputs.
     - Metadata summaries.
     - Contextual information from retrieved documents.
- **Workflow**:
   - Receives preprocessed data from the Metadata Summarizer and Vectorstore Retriever.
   - Generates coherent, detailed, and context-aware responses.
- **Example**:
   - Input: “What are the best features of this product?”
   - Output: "The product offers industry-leading noise cancellation, lightweight construction, and a battery life of 10 hours."
- **LangFuse Integration**:
   - Tracks token usage and API credits consumed during response generation.

---

### **6. Follow-up Question Generator**
- **Role**:
   - Enhances the user experience by generating relevant follow-up questions.
- **Workflow**:
   - Evaluates the interaction context and the generated response.
   - Suggests clarifying or exploratory follow-up queries.
- **Example**:
   - "Would you like me to compare this product with similar options?"
- **LangFuse Integration**:
   - Logs generated follow-up questions and engagement metrics.

---

### **7. LangFuse Analytics**
- **Role**:
   - Monitors and logs operational metrics across the entire pipeline.
- **Features**:
   - **Trace Logging**: Tracks module inputs, outputs, and execution times.
   - **Token Monitoring**: Logs token consumption for summarization, retrieval, and response generation.
   - **Cost Attribution**: Provides insights into the costs of API interactions and token usage.
- **Use Case**:
   - Analyzing the pipeline’s efficiency and optimizing cost-to-performance ratios.

---

## **Pipeline Execution Workflow**

1. **User Query Submission**  
   - Users interact with the chatbot via the frontend interface, which sends queries to the backend through the `dev/stream` API.

2. **Supervisor Routing**  
   - The Supervisor Module determines whether the query relates to structured metadata or unstructured contextual data.

3. **Data Retrieval**  
   - Metadata queries → Metadata Retriever fetches structured data.
   - Contextual queries → Vectorstore Retriever retrieves relevant unstructured data.

4. **Metadata Summarization**  
   - Summarizes structured metadata for clarity and brevity.

5. **Response Generation**  
   - The Main LLM combines all inputs to generate a coherent and relevant response.

6. **Follow-up Engagement**  
   - A follow-up query is generated to improve user interaction and clarify ambiguous queries.

7. **Analytics Logging**  
   - LangFuse logs the entire interaction, including token usage, trace data, and performance metrics.

---

## **API Integration**

### **Request Example**
```json
{
  "query": "What are the features of this product?",
  "parent_asin": "B08K2S3D2K",
  "user_id": "user_12345",
  "log_langfuse": true,
  "stream_tokens": true
}
```

### **Response Example**
```json
{
  "response": "This product features noise cancellation, long battery life, and a sleek design.",
  "follow_up": "Would you like me to compare this product with similar options?"
}
```
