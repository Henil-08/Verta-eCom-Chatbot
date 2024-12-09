# Bias Detection Pipeline Documentation

This document outlines the workflow, components, and functionality of the `bias_detection` pipeline for the **Verta Chatbot**. It provides detailed steps to guide users and developers in understanding and leveraging the pipeline for detecting bias in product reviews.

## **Overview**

The `bias_detection` pipeline detects biases in product reviews by combining sentiment analysis, embeddings, and metadata analysis. The detection criteria include over-reliance on negative reviews, lack of acknowledgment of sparse data, and imbalanced sentiment representation. The pipeline supports modular experimentation by allowing parameter and configuration adjustments in the config/ directory.

---

## **Pipeline Steps**

### 1. **Initialization**

The `BiasDetection` class is initialized with the following:
- **`BiasDetectionConfig`**: Defines paths, MLFlow configurations, and bias detection models and parameters.

### 2. **Test Data Loading**

The **`read_parquet`** function:
- Reads test datasets from the `evaluation/results/` directory (Parquet format).

### 3. **Product Reviews Retrieval**

The **`load_product_reviews`** function:
- Connects to the database using credentials stored as environment variables.
- Fetches product reviews from SQL tables:
  - **`userreviews`**: Contains customer reviews (text, helpful votes, verified purchase).

### 4. **Sentiment Analysis on Reviews**

The **`analyze_sentiments`** function:
- For each product review, calls the sentiment analyzer model:
  - Classifies each review as positive, neutral, or negative.
  - Returns a count of how many reviews fall into each sentiment category.

### 5. **Sentiment Probability Analysis of the Response**

The **`analyze_sentiments_with_probs`** function:
- Given the final response (model-generated answer), uses the probability sentiment analyzer:
  - Expects a JSON-formatted string with keys for positive, neutral, and negative probabilities.
  - Returns a dictionary mapping each sentiment to its predicted probability.

### 6. **Sparse Data Acknowledgment Check**

The **`sparse_data_acknowledged`** function:
- Given the response
  - Generates the embeddings using HuggingFaceEmbeddings
  - Compares this embedding with embeddings of predefined SPARSE_DATA_PHRASES.
  - Computes cosine similarities and checks if the maximum similarity exceeds a threshold (0.6).
  - Returns True if the response acknowledges sparse data conditions, else False.


### 6. **Bias Detection**

The **`bias_detection`** function:
- Given the responses, counter for sentiments on review data and number of reviews:
  - Uses the probabilities of negative sentiment in the final response.
  - Checks the ratio of negative to positive reviews for the product.
    - If the final response is strongly negative (>0.7 probability) and the underlying reviews are indeed more negative than positive, flags potential bias of "over_reliance_on_negative".
    - If the number of reviews is small (<4) and the response does not acknowledge sparse data, flags "missing_data_acknowledgment".
  - Returns a dictionary indicating if bias was detected and which types of bias were found.


### 7. **Running the Full Detection Process**

The **`detect`** function:
  - Iterates over each row in the evaluation results (each containing a response and asin):
  - Loads the corresponding product reviews and determines their sentiment distribution.
  - Runs bias_detection on the response and stores results per asin.
  - Aggregates results across all ASINs.

### 8. **Result Saving**

The **`save_score`** function:
- Persists bias detection results in **Parquet** and **JSON** formats for downstream analysis.


---

## **Key Configurations**

- **File Paths**:
  - `embedding_model`: Embedding model used to get the embeddings of text.
  - `sentiment_model`: Sentiment model used to do sentiment analysis for the text.
  - `prompt_sentiment`: Prompt used to generate the sentiment analysis.
  - `prompt_prob_sentiment`: Prompt used to generate the sentiment analysis porbablities for each sentiment.
  - `results_path`: Directory to save bias detection results.

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
   python src/pipeline/stage_04_bias_detection.py
   ```

3. **Monitor Results**:
   - Check generated responses in the `results/` directory.
   - View metrics in the `evaluation/metrics/bias-scores.json` file.

4. **Experimentation**:
   - Update the prompt, embeddings, or retrieval strategy in the `config/` directory to experiment with different models.

---

## **Metrics and Insights**

| Metric                   | Description                                                                                    |
|--------------------------|------------------------------------------------------------------------------------------------|
| **bias detected count**  | Tracks the total number of responses for a specific ASIN that have been flagged as biased.     |
| **bias types**           | A set of distinct bias types identified for a specific ASIN.                                   |
| **num reviews**          | total number of product reviews retrieved and analyzed for each ASIN.                          |
| **review sentiments**    | A breakdown of the sentiment distribution (positive, neutral, negative).                       |
---

## **Troubleshooting**

- **Database Connection Errors**:
  Ensure all environment variables are correctly set for database credentials.

- **Missing Data**:
  Verify the presence of Parquet files in `evaluation/testset/base-results.parquet` directory.

- **Incorrect Model Names**:
  If embedding or sentiment models fail to load, the pipeline uses fallback models and logs a warning.
---

## **Future Improvements**
- **Refined Bias Criteria**:
  Add more nuanced checks for bias based on additional sentiment categories or linguistic features.

- **Scalability**:
  Handle larger datasets more efficiently or integrate caching.

- **Error Handling**:
Improve robustness to handle cases where certain products have no reviews or where database queries fail.

--- 

## **Acknowledgments**

This pipeline leverages the following libraries and tools:

  - **HuggingFace Transformers**: For embeddings and sentiment analysis.
  - **SQLAlchemy**: For database connectivity.