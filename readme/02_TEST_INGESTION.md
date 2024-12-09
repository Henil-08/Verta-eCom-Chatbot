# Test Ingestion Pipeline Documentation

This document outlines the workflow, components, and functionality of the `test_ingestion` pipeline for the **Verta Chatbot**. It provides detailed steps to guide users and developers in understanding and leveraging the pipeline for synthetic test case generation.

## Overview

The `TestIngestion` pipeline is a core component of the Verta Chatbot system, designed to fetch product data (reviews and metadata), generate summaries and test datasets, and store them in a structured format for further use. This documentation outlines the steps involved in the pipeline, the configuration files required, and the methodology used.

---

## **Directory Structure**

```
root/
├── config/
    ├── config.yaml              # Defines parameters for all the pipelines.
    ├── prompts.yaml            # Stores the LLM prompts for generating responses.
├── cache/
    ├── faiss/                  # Directory for storing FAISS vector databases.
    ├── metadata/               # Stores product metadata in CSV format.
├── evaluation/
    ├── metrics/                # Stores evaluation metrics.
    ├── results/                # Stores evaluation results as parquets.
    ├── testset/                # Contains test datasets in Parquet format.
```

---

## Pipeline Steps

### 1. **Initialization**
The `TestIngestion` class is initialized with:
- **`config`**: A `TestIngestionConfig` object containing paths and prompts.
- **`product_asins`**: A list of product ASINs for which data will be processed.

The initialization step also generates a unique UUID for each ASIN to ensure traceability.

### 2. **Load Product Data**
The `load_product_data(asin)` function:
- Connects to a database using credentials from environment variables.
- Fetches:
  - **Reviews**: Data about user reviews for the product.
  - **Metadata**: Product details, including category, rating, price, and features.


### 3. **Generate Metadata Summary**
The `generate_meta_summary(meta_df, prompt)` function:
- Utilizes the **ChatGroq** LLM to create a concise summary of product metadata.
- Steps:
  - Format metadata details and replace unsupported characters.
  - Construct a system prompt using the metadata.
  - Generate a response through a `ChatPromptTemplate`.


### 4. **Create Documents**
The `create_docs(review_df, meta_df)` function:
- Converts review data into **LangChain documents** using `DataFrameLoader`.
- Prepends a metadata summary document to the list of review documents.


### 5. **Generate RAGAS Testset**
The `get_ragas_testset(doc, with_debugging_logs=False)` function:
- Uses:
  - **Generator LLM**: GPT-based LLM for generating test questions.
  - **Critic LLM**: GPT-based LLM for evaluating test questions.
  - **Embeddings**: `OpenAIEmbeddings` for context embedding.
- Generates a test set based on document data, adhering to specified distributions:
  - **Simple**: 50%
  - **Reasoning**: 25%
  - **Multi-Context**: 25%


### 6. **Transform Testset**
The `transform_ragas_testset_df(testset, asin)` function:
- Converts the test set into a Pandas DataFrame.
- Adds a column for `parent_asin`.
- Drops unused columns like `metadata` and `episode_done`.


### 7. **Save UUIDs and Testsets**
The `save_hash(hash)` function:
- Saves ASIN-to-UUID mapping as a JSON file in the artifact directory.

The `generate_test_set()` function:
- Iterates through the ASINs to process each product:
  1. Fetch product data (reviews and metadata).
  2. Generate documents and summaries.
  3. Create and save test sets as Parquet files.

---

## **Key Configurations**

- **File Paths**:
  - `testset_path`: Location to save test datasets.
  - `faiss_dir`: Directory for FAISS vector stores.
  - `meta_dir`: Directory for metadata files.

- **Database Connection**:
  - Environment variables for DB credentials:
    - `INSTANCE_CONNECTION_NAME`
    - `DB_USER`
    - `DB_PASS`
    - `DB_NAME`

---

## **How to Use**

1. **Set Up Configurations**:
   - Modify `config.yaml` and `prompts.yaml` in the `config/` directory to customize parameters.

2. **Run the Pipeline**:
   Execute the following command:
   ```bash
   python src/pipeline/stage_02_test_ingestion.py
   ```

3. **Monitor Results**:
   - Check generated responses in the `evaluate/testset/` directory.
   - View uuid hash file in the `artifacts/product_uuids.json` file.

---

## **Troubleshooting**

- **Database Connection Errors**:
  Ensure all environment variables are correctly set for database credentials.

- **Missing Data**:
  Verify the presence of Parquet files in `evaluation/testset/product_uuids.json` directory.

- **MLFlow Issues**:
  Confirm the tracking URI and MLFlow configuration in the `config.yaml` file.

---

## **Acknowledgments**

This pipeline leverages the following libraries and tools:
- **LangGraph** for the state graph.
- **RAGAS** for synthetic test case generation.
- **FAISS** for vector storage.