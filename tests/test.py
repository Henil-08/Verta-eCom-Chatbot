import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import pandas as pd
import pytest
import sqlalchemy
from fastapi import status
from unittest.mock import MagicMock, Mock
from httpx import AsyncClient
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from src.components.agents import metadata_node, retrieve, supervisor_agent
from src.serve import app
from src.components.nodes import final_llm_node, followup_node, route_question
from src.config.configuration import ConfigurationManager
from src.utils.database import connect_with_db

# load the API Keys
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["LANGFUSE_PUBLIC_KEY"] = os.getenv("LANGFUSE_PUBLIC_KEY")
os.environ["LANGFUSE_SECRET_KEY"] = os.getenv("LANGFUSE_SECRET_KEY")
os.environ["LANGFUSE_HOST"] = os.getenv("LANGFUSE_HOST")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
credentials = {
    "INSTANCE_CONNECTION_NAME": os.getenv("INSTANCE_CONNECTION_NAME"),
    "DB_USER": os.getenv("DB_USER"),
    "DB_PASS": os.getenv("DB_PASS"),
    "DB_NAME": os.getenv("DB_NAME"),
}

VERTA_API_ACCESS_TOKEN = os.environ["VERTA_API_ACCESS_TOKEN"]
INVALID_TOKEN = "mock_invalid_token"

configManager = ConfigurationManager()
config = configManager.get_prepare_base_model_config()

@pytest.fixture
def valid_headers():
    """Provide valid authentication headers for tests."""
    return {"Authorization": f"Bearer {VERTA_API_ACCESS_TOKEN}"}


@pytest.fixture
def invalid_headers():
    """Provide invalid authentication headers for tests."""
    return {"Authorization": f"Bearer {INVALID_TOKEN}"}


@pytest.fixture
def test_payload():
    """Provide a sample payload for POST requests."""
    return {
        "query": "Hello",
        "parent_asin": "B072K6TLJX",
        "user_id": "ABCD",
        "log_langfuse": 1,
        "stream_tokens": 1,
    }


# Mock database connection
@pytest.fixture
def mock_db_connection() -> None:
    mock_conn = MagicMock()
    mock_conn.execute.return_value.fetchall.return_value = [
        (
            "B07MQFWF1B",
            "asin1",
            10,
            "2024-01-01",
            True,
            "Test Product",
            "Great product!",
        )
    ]
    return mock_conn


# Test the database connection
def test_connect_with_db() -> None:
    engine = connect_with_db(credentials=credentials)
    assert isinstance(engine, sqlalchemy.engine.base.Engine)


# Test supervisor agent logic
def test_supervisor_agent() -> None:
    state = {
        "question": "What is the product price?",
        "question_type": "",
        "answer": "The product price is $20.",
        "documents": [],
        "meta_data": pd.DataFrame(),
        "retriever": MagicMock(),
        "followup_questions": [],
    }

    result = supervisor_agent(
        state, prompt=config.prompt_supervisor, model=config.supervisor_model
    )
    assert "question_type" in result


# Test route_question logic
def test_route_question():
    state = {
        "question": "What is the product price?",
        "question_type": Mock(datasource="Review-Vectorstore"),
        "answer": "The product price is $20.",
        "documents": [],
        "meta_data": pd.DataFrame(),
        "retriever": MagicMock(),
        "followup_questions": [],
    }

    result = route_question(state)
    assert result in ["Review-Vectorstore", "FINISH"]


# Test metadata_node logic
def test_metadata_node():
    state = {
        "question": "What is the product description?",
        "question_type": "Metadata",
        "answer": "",
        "documents": [],
        "meta_data": pd.DataFrame(
            {
                "main_category": ["Electronics"],
                "title": ["Test Product"],
                "average_rating": [4.5],
                "rating_number": [100],
                "features": ["Great feature"],
                "description": ["This is a test product description."],
                "price": [20],
                "store": ["Test Store"],
                "categories": ["Category1"],
                "details": ["Detailed information about the product."],
            }
        ),
        "retriever": MagicMock(),
        "followup_questions": [],
    }

    result = metadata_node(
        state, prompt=config.prompt_metadata, model=config.metadata_model
    )
    assert isinstance(result, dict)
    assert "meta_summary" in result


# Test retrieve function
def test_retrieve():
    docs = [Document(page_content="This is a review.")]
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectordb = FAISS.from_documents(documents=docs, embedding=embeddings)
    retriever = vectordb.as_retriever()

    state = {
        "question": "What is the product review?",
        "question_type": "Review-Vectorstore",
        "answer": "",
        "documents": [],
        "meta_data": pd.DataFrame(),
        "retriever": retriever,
        "followup_questions": [],
    }

    result = retrieve(state)
    assert "documents" in result


# Test final_llm_node function
def test_final_llm_node():
    state = {
        "question": "Tell me more about the product.",
        "meta_summary": Document(page_content="Product details."),
        "documents": [Document(page_content="Product details.")],
        "answer": "This product is amazing!",
    }

    result = final_llm_node(
        state, prompt=config.prompt_base_model, model=config.base_model
    )
    assert "answer" in result
    assert result["answer"].content


# Test followup_node function
def test_followup_node():
    state = {
        "question": "Tell me more about the product.",
        "answer": Mock(content="This product is amazing!"),
        "meta_summary": Document(page_content="Product details."),
        "documents": [Document(page_content="Product details.")],
    }

    result = followup_node(
        state, prompt=config.prompt_followup, model=config.followup_model
    )
    assert isinstance(result, dict)
    assert "followup_questions" in result


@pytest.mark.asyncio
async def test_health_endpoint():
    """Test the health check endpoint."""
    async with AsyncClient(app=app, base_url="http://localhost:80") as client:
        response = await client.get("/")
        assert response.status_code == status.HTTP_200_OK
        assert response.json() == {"status": "🤙"}


@pytest.mark.asyncio
async def test_initialize_valid_token(valid_headers):
    """Test the initialize endpoint with a valid token."""
    async with AsyncClient(app=app, base_url="http://localhost:80") as client:
        response = await client.get(
            "/initialize",
            params={"asin": "B072K6TLJX", "user_id": 123},
            headers=valid_headers,
        )
        assert response.status_code == status.HTTP_200_OK
        assert response.json()["status"] == "retriever initialized"


@pytest.mark.asyncio
async def test_initialize_invalid_token(invalid_headers):
    """Test the initialize endpoint with an invalid token."""
    async with AsyncClient(app=app, base_url="http://localhost:80") as client:
        response = await client.get(
            "/initialize",
            params={"asin": "B072K6TLJX", "user_id": 123},
            headers=invalid_headers,
        )
        assert response.status_code == status.HTTP_401_UNAUTHORIZED
        assert response.json()["detail"] == "Invalid or expired token"


@pytest.mark.asyncio
async def test_score_endpoint(valid_headers):
    """Test the score endpoint."""
    async with AsyncClient(app=app, base_url="http://localhost:80") as client:
        payload = {
            "parent_asin": "B072K6TLJX",
            "user_id": "ABCD",
            "run_id": "334b57ee-1a9e-426b-860c-6a6f3792cb55",
            "value": 1,
        }
        response = await client.post("/score", json=payload, headers=valid_headers)
        assert response.status_code == status.HTTP_200_OK


@pytest.mark.asyncio
async def test_invoke_endpoint(valid_headers, test_payload):
    """Test the invoke endpoint."""
    async with AsyncClient(app=app, base_url="http://localhost:80") as client:
        response = await client.post(
            "/dev-invoke", json=test_payload, headers=valid_headers
        )
        assert response.status_code in {status.HTTP_200_OK, status.HTTP_400_BAD_REQUEST}


@pytest.mark.asyncio
async def test_stream_endpoint(valid_headers, test_payload):
    """Test the stream endpoint for server-sent events."""
    async with AsyncClient(app=app, base_url="http://localhost:80") as client:
        async with client.stream(
            "POST", "/dev-stream", json=test_payload, headers=valid_headers
        ) as response:
            assert response.status_code == status.HTTP_200_OK
            # Read the stream (partial data expected)
            async for line in response.aiter_text():
                assert line.startswith("data:")


@pytest.mark.asyncio
async def test_invalid_stream_endpoint(invalid_headers, test_payload):
    """Test the stream endpoint with an invalid token."""
    async with AsyncClient(app=app, base_url="http://localhost:80") as client:
        response = await client.post(
            "/dev-stream", json=test_payload, headers=invalid_headers
        )
        assert response.status_code == status.HTTP_401_UNAUTHORIZED