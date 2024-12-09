# API Documentation for Verta Chatbot

This API is designed to handle user input through Verta and return responses, with support for both streaming and non-streaming modes. We used FastAPI to deploy the Verta REST API to GCP Cloud Run so it automatically comes with the models at the /docs route e.g. https://verta-chat-service-269431978711.us-east1.run.app/docs

You can use this interface to see & test all API endpoints.

---

## Base URL

```
https://verta-chat-service-269431978711.us-east1.run.app
```

---

## Authorization
The API has user protected endpoints that require a bearer token that we have assigned and will Slack to you.

---

## Endpoints

### 1. `/`

**Method:** `GET`
**Description:** Root endpoint; health check of API, returns 
```json
{
  "status": "🤙"
} 
```
when service is up; not user protected


### 2. `/initialize`

**Method:** `GET`

**Description:** Initializes the vector store retriever for a specific product (ASIN) and user ID. This process caches the review and metadata for efficient future queries.

**Query Parameters:**

| Parameter | Type | Description |
| --- | --- | --- |
| asin | string | The product's Parent ASIN ID. |
| user_id | int | The unique identifier for the user. |

**Response:**

```json
{
  "status": "retriever initialized",
  "asin": "string",
  "user_id": "integer"
}

```


### 3. `/score`

**Method:** `POST`

**Description:** Adds User-Feedback for a specified product (ASIN) and user ID.

**Request Body:**

```json
{
    "run_id": "string",
    "parent_asin": "string",
    "user_id": "string",
    "value": true
}

```

| Field | Type | Description |
| --- | --- | --- |
| run_id | string | The message ID for the response from LLM. |
| parent_asin | string | The product's Parent ASIN ID. |
| user_id | string | The unique identifier for the user. |
| value | boolean | User Feedback as 0 or 1. |

**Response:**

```json
{ 
  "status": "Feedback Successful", 
  "trace_id": "trace_id", 
  "id": "id"
}

```

Returns **500 Bad Request** if failed to add user-feedback


### 4. `/dev-invoke`

**Method:** `POST`

**Description:** Invokes the agent with the given user input and retrieves a complete response (non-streaming).

**Request Body:**

```json
{
  "query": "string",
  "parent_asin": "string",
  "user_id": "string",
  "log_langfuse": true,
  "stream_tokens": false
}
```

| Field          | Type   | Description                                                        |
|----------------|--------|--------------------------------------------------------------------|
| query     | string | The question or query you want the agent to respond to.            |
| parent_asin    | string | The Parent Asin Id of the product querying.                        |  
| user_id        | string | The User-ID of the user logged in.                                 |
| log_langfuse   | bool   | Whether to log responses and interactions to Langfuse.             |
| stream_tokens  | bool   | If true, streaming tokens are used; otherwise, a full response is returned. (No use in invoke method)|

**Response:**

```json
{
  "run_id": "string",
  "question": "string",
  "answer": "string",
  "followup_questions": [
    "string",
    "string",
    "string"
  ]
}
```

| Field             | Type   | Description                                                             |
|-------------------|--------|-------------------------------------------------------------------------|
| run_id            | string | The Id for the response.                                                |
| question          | string | The user query or question submitted.                                   |
| answer            | string | The agent's full response.                                              |
| followup_questions| array  | Suggested follow-up questions based on the answer.                      |

### 5. `/dev-stream`

**Method:** `POST`

**Description:** Streams the agent's response to a user input, including intermediate messages and tokens.

**Request Body:**

```json
{
  "user_input": "string",
  "parent_asin": "string",
  "user_id": "string",
  "log_langfuse": true,
  "stream_tokens": true
}
```

| Field          | Type   | Description                                                        |
|----------------|--------|--------------------------------------------------------------------|
| user_input     | string | The question or query you want the agent to respond to.            |
| parent_asin    | string | The Parent Asin Id of the product querying.                        |  
| user_id        | string | The User-ID of the user logged in.                                 |
| log_langfuse   | bool   | Whether to log responses and interactions to Langfuse.             |
| stream_tokens  | bool   | If true, token-by-token responses are streamed.                    |

**Response:**

Streams intermediate responses and tokens (if `stream_tokens` is set to `True`). The final message contains the full answer and associated citations.

Example:

``` json
data: {"type": "token", "content": "Hello "}
data: {"type": "token", "content": "World"}
data: {"type": "message", "content": {
    "run_id" : "string",
    "question": "string",
    "answer": "string",
    "followup_questions": [
      "string",
      "string",
      "string"
    ]
  }
}
data: [DONE]
```

---

### Usage Examples

**Invoke Example (cURL):**

```bash
curl -X POST "http://localhost:80/invoke" -H "Content-Type: application/json" -d '{
  "user_input": "Explain how to motivate a grade 2 student.",
  "config": {"thread_id": "2"},
  "log_langfuse": 1,
  "stream_tokens": 0
}'
```

**Stream Example (cURL):**

```bash
curl -X POST "http://localhost:80/stream" -H "Content-Type: application/json" -d '{
  "user_input": "How to plan a lesson on fractions?",
  "config": {"thread_id": "2"},
  "log_langfuse": 1,
  "stream_tokens": 1
}' --no-buffer
```

---

### Installation and Setup

Refer to the main `README.md` for installation and setup instructions.

---

### Notes

- Ensure that environment variables for `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, and `LANGFUSE_HOST` are properly set.
- For tracing and monitoring, Langfuse integration is used for logging interactions and responses.

---

### Error Codes and Messages

- **400:** FastAPI returns a 400 status code for generic client errors
- **500:** FastAPI returns a 500 status code for server errors, such as when there's a Pydantic ValidationError in your code. This prevents clients from seeing internal information about the error, which could be a security vulnerability.