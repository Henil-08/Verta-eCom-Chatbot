import os
import uuid
import json
import shutil
import asyncio
import uvicorn
import pandas as pd
from src import logger
from typing import Any
from pathlib import Path
from sqlalchemy import text
from langfuse import Langfuse
from pydantic import BaseModel
from src.old_graph import create_graph
from src.utils import database as db
from datetime import datetime, timedelta
from contextlib import asynccontextmanager
from collections.abc import AsyncGenerator
from langfuse.callback import CallbackHandler
from fastapi.middleware.cors import CORSMiddleware
from langchain_community.vectorstores import FAISS
from langgraph.graph.state import CompiledStateGraph
from langchain_huggingface import HuggingFaceEmbeddings
from fastapi.responses import StreamingResponse, JSONResponse
from langchain_community.document_loaders import DataFrameLoader
from fastapi import FastAPI, HTTPException, status, Query, Request

from dotenv import load_dotenv
load_dotenv()

# Define the directory paths
cache_dir = Path("cache")
faiss_dir = cache_dir / "faiss"
meta_dir = cache_dir / "meta"

# Create directories if they don't exist
for directory in [cache_dir, faiss_dir, meta_dir]:
    directory.mkdir(parents=True, exist_ok=True)

## load the API Keys
os.environ['HF_TOKEN']=os.getenv("HF_TOKEN")
os.environ['OPENAI_API_KEY']=os.getenv("OPENAI_API_KEY")
os.environ['GROQ_API_KEY']=os.getenv("GROQ_API_KEY")

## Langfuse
os.environ['LANGFUSE_PUBLIC_KEY']=os.getenv("LANGFUSE_PUBLIC_KEY")
os.environ['LANGFUSE_SECRET_KEY']=os.getenv("LANGFUSE_SECRET_KEY")
os.environ['LANGFUSE_HOST']=os.getenv("LANGFUSE_HOST")

## Postgres DB
credentials = {
    'INSTANCE_CONNECTION_NAME': os.getenv("INSTANCE_CONNECTION_NAME"),
    'DB_USER': os.getenv("DB_USER"),
    'DB_PASS': os.getenv("DB_PASS"),
    'DB_NAME': os.getenv("DB_NAME")
}

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

vector_store_cache = [] # a list to track the cache
cache_creation_times = {} # a dictionary to track cache creation times

engine = db.connect_with_db(credentials)
langfuse = Langfuse()

class UserInput(BaseModel):
    user_input: str
    parent_asin: str
    user_id: str
    log_langfuse: bool
    stream_tokens: bool

class clearCache(BaseModel):
    user_id: str
    parent_asin: str


class scoreTrace(BaseModel):
    run_id: str
    user_id: str
    parent_asin: str
    value: bool


@app.middleware("http")
async def log_requests(request: Request, call_next):
    # Start time
    start_time = datetime.utcnow()
    response = await call_next(request)

    # End time
    process_time = (datetime.utcnow() - start_time).total_seconds()
    
    log_data = {
        "timestamp": datetime.utcnow().isoformat(),
        "method": request.method,
        "url": str(request.url),
        "status_code": response.status_code,
        "process_time": process_time,
        "client_ip": request.client.host
    }
    
    # Send log to Google Cloud Logging or stdout for Cloud Run to capture
    logger.info(json.dumps(log_data))
    return response


# Define a background task to delete cache files older than 1 hour
async def clear_outdated_cache():
    while True:
        current_time = datetime.utcnow()
        for cache_key, creation_time in list(cache_creation_times.items()):
            if (current_time - creation_time) > timedelta(hour=1):
                try:
                    # Remove cache files
                    if os.path.exists(f"{faiss_dir}/{cache_key}") and os.path.isdir(f"{faiss_dir}/{cache_key}"):
                        shutil.rmtree(f"{faiss_dir}/{cache_key}")
                    if os.path.exists(f"{meta_dir}/{cache_key}.csv") and os.path.isfile(f"{meta_dir}/{cache_key}.csv"):
                        os.remove(f"{meta_dir}/{cache_key}.csv")

                    # Remove from cache tracking
                    vector_store_cache.remove(cache_key)
                    del cache_creation_times[cache_key]

                    logger.info(f"Cache automatically cleared for {cache_key}")
                except Exception as e:
                    logger.error(f"Error clearing cache for {cache_key}: {e}")

        # Wait an hour before the next check
        await asyncio.sleep(3600)


# Use lifespan to manage startup and shutdown events
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup tasks
    cache_task = asyncio.create_task(clear_outdated_cache())
    try:
        yield
    finally:
        cache_task.cancel()
        await cache_task  # Wait for the task to be cancelled

app.router.lifespan_context = lifespan


async def load_product_data(asin: str):
    with engine.begin() as connection:
        try:
            # Log start of data fetching
            logger.info(f"Loading product data for ASIN: {asin}")

            # Fetch reviews
            review_query = text(f"""
                SELECT parent_asin, asin, helpful_vote, timestamp, verified_purchase, title, text
                FROM userreviews ur 
                WHERE ur.parent_asin = '{asin}';
            """)
            review_result = connection.execute(review_query)
            review_df = pd.DataFrame(review_result.fetchall(), columns=review_result.keys())
            logger.info("Fetched review data")

            # Fetch metadata
            meta_query = text(f"""
                SELECT parent_asin, main_category, title, average_rating, rating_number, features, description, price, store, categories, details
                FROM metadata md 
                WHERE md.parent_asin = '{asin}';
            """)
            meta_result = connection.execute(meta_query)
            meta_df = pd.DataFrame(meta_result.fetchall(), columns=meta_result.keys())
            logger.info("Fetched metadata")

        except Exception as e:
            logger.error(f"Error loading data for ASIN: {asin} - {e}")
            raise HTTPException(status_code=500, detail="Error loading data")

    return review_df, meta_df


def create_vector_store(review_df):
    logger.info("Creating vector store from review data")
    loader = DataFrameLoader(review_df)
    review_docs = loader.load()

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectordb = FAISS.from_documents(documents=review_docs, embedding=embeddings)
    logger.info("Vector store created successfully")
    return vectordb


@app.get("/")
async def root():
    return {"message": "Welcome to the Verta FastAPI app!"}


@app.get("/initialize")
async def initialize(asin: str = Query(...), user_id: int = Query(...)):
    logger.info(f"Received request to initialize retriever for ASIN: {asin} and User ID: {user_id}")
    
    cache_key = f"{user_id}-{asin}"

    if cache_key not in vector_store_cache:
        review_df, meta_df = await load_product_data(asin)
        vector_db = create_vector_store(review_df)
        vector_db.save_local(f"{faiss_dir}/{cache_key}")
        meta_df.to_csv(f"{meta_dir}/{cache_key}.csv", index=False)
        vector_store_cache.append(cache_key)
        cache_creation_times[cache_key] = datetime.utcnow()

        logger.info(f"Retriever initialized and cached for ASIN: {asin} and User ID: {user_id}")
    else:
        logger.info(f"Retriever for ASIN: {asin} and User ID: {user_id} already cached")

    return JSONResponse(content={"status": "retriever initialized", "asin": asin, "user_id": user_id}, status_code=200)


@app.post("/score")
async def annotate(score_trace: scoreTrace):
    trace_id = score_trace.run_id
    user_id = score_trace.user_id
    asin = score_trace.parent_asin
    id = str(uuid.uuid4()) + f"-{user_id}-{asin}"
    value = score_trace.value
    try:
        langfuse.score(
            id=id,
            trace_id=trace_id,
            name="user-feedback",
            value=value,
            data_type="BOOLEAN" 
        )
        logger.info(f"Feedback Successful, 'trace_id': {trace_id}, 'id': {id}")
        return JSONResponse(content={"status": "Feedback Successful", "trace_id": trace_id, "id": id}, status_code=200)
    except Exception as e:
        logger.error(f"Error Scoring Trace: {trace_id} - {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/dev-invoke")
async def invoke(user_input: UserInput):
    cache_key = f"{user_input.user_id}-{user_input.parent_asin}"
    logger.info(f"Received request to invoke agent for ASIN: {user_input.parent_asin} and User ID: {user_input.user_id}")

    if cache_key not in vector_store_cache:
        review_df, meta_df = await load_product_data(user_input.parent_asin)
        vector_db = create_vector_store(review_df)
        vector_db.save_local(f"{faiss_dir}/{cache_key}")
        meta_df.to_csv(f"{meta_dir}/{cache_key}.csv", index=False)
        vector_store_cache.append(cache_key)
        cache_creation_times[cache_key] = datetime.utcnow()
        logger.info(f"Cache initialized for ASIN: {user_input.parent_asin} and User ID: {user_input.user_id}")

    # Ensure paths exist
    retriever = f"{faiss_dir}/{cache_key}"
    meta_df = f"{meta_dir}/{cache_key}.csv"

    if not os.path.exists(retriever):
        logger.error(f"Retriever not initialized for ASIN: {user_input.parent_asin} and User ID: {user_input.user_id}")
        return JSONResponse(content={"status": "Retriever not initialized"}, status_code=400)
    if not os.path.exists(meta_df):
        logger.error(f"Meta-Data not initialized for ASIN: {user_input.parent_asin} and User ID: {user_input.user_id}")
        return JSONResponse(content={"status": "Meta-Data not initialized"}, status_code=400)

    agent: CompiledStateGraph = create_graph()
    config = {"configurable": {"thread_id": f"{cache_key}"}}

    if user_input.log_langfuse:
        run_id = str(uuid.uuid4())
        langfuse_handler = CallbackHandler(
            user_id=f"{user_input.user_id}",
            session_id=f"{cache_key}"
        )
        config.update({"callbacks": [langfuse_handler], "run_id": run_id})
    try:
        response = agent.invoke({
            "question": user_input.user_input, 
            "meta_data": meta_df,
            "retriever": retriever
        }, config=config)
        logger.info(f"Agent response generated for User ID: {user_input.user_id}")

        output = {
            'run_id': run_id,
            'question': response['question'],
            'answer': response['answer'].content,
            'followup_questions': response['followup_questions']
        }
        logger.debug(f"Final response: {output}")
        return output
    except Exception as e:
        logger.error(f"Error invoking agent for User ID: {user_input.user_id} - {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def message_generator(user_input: UserInput, stream_tokens=True) -> AsyncGenerator[str, None]:
    """
    Generate a stream of messages from the agent.

    This is the workhorse method for the /stream endpoint.
    """
    cache_key = f"{user_input.user_id}-{user_input.parent_asin}"
    logger.info(f"Initializing message generator for cache key: {cache_key}")

    if cache_key not in vector_store_cache:
        logger.info(f"Cache miss for {cache_key}. Loading product data and initializing vector store.")
        review_df, meta_df = await load_product_data(user_input.parent_asin)
        vector_db = create_vector_store(review_df)
        vector_db.save_local(f"{faiss_dir}/{cache_key}")
        meta_df.to_csv(f"{meta_dir}/{cache_key}.csv", index=False)
        vector_store_cache.append(cache_key)
        cache_creation_times[cache_key] = datetime.utcnow()
        logger.info(f"Vector store and metadata initialized and cached for {cache_key}")

    # Ensure paths exist
    retriever = f"{faiss_dir}/{cache_key}"
    meta_df = f"{meta_dir}/{cache_key}.csv"

    if not os.path.exists(retriever):
        logger.error(f"Retriever not initialized for cache key: {cache_key}")
        yield JSONResponse(content={"status": "Retriever not initialized"}, status_code=400)
    if not os.path.exists(meta_df):
        logger.error(f"Metadata file not found for cache key: {cache_key}")
        yield JSONResponse(content={"status": "Meta-Data not initialized"}, status_code=400)
    
    agent: CompiledStateGraph = create_graph()
    config = {"configurable": {"thread_id": f"{user_input.user_id}"}}
    if user_input.log_langfuse:
        run_id = str(uuid.uuid4())
        langfuse_handler = CallbackHandler(
            user_id=f"{user_input.user_id}",
            session_id=f"{cache_key}"
        )
        config.update({"callbacks": [langfuse_handler], "run_id": run_id})
    if user_input.stream_tokens == 0:
        stream_tokens = False

    logger.info("Starting event stream processing for agent.")
    
    # Process streamed events from the graph and yield messages over the SSE stream.
    async for event in agent.astream_events({"question": user_input.user_input, 
                                            "meta_data": meta_df,
                                            "retriever": retriever
                                        }, version="v2", config=config):
        if not event:
            logger.warning("Received empty event in stream.")
            continue

        # Yield tokens streamed from LLMs.
        if (
            event["event"] == "on_chat_model_stream"
            and stream_tokens == True
            and any(t.startswith('seq:step:2') for t in event.get("tags", []))
            and event['metadata']['langgraph_node'] == 'generate'
        ):
            content = event["data"]["chunk"].content
            if content:
                logger.debug(f"Streaming token: {content}")
                yield f"data: {json.dumps({'type': 'token', 'content': content})}\n\n"
            continue

        # Yield messages written to the graph state after node execution finishes.
        if (
            (event["event"] == "on_chain_end")
            and ((any(t.startswith('seq:step:2') for t in event.get("tags", [])))
            and ((event['metadata']['langgraph_node'] == 'final')
            and (event['metadata']['langgraph_triggers'] == ['generate'])))
        ):
            answer = event["data"]["output"]["answer"].content
            followup_questions = event["data"]["output"]["followup_questions"]
            output = {
                'run_id': run_id,
                "question": user_input.user_input,
                "answer": answer,
                "followup_questions": followup_questions
            }
            logger.info(f"Yielding final response for question: {user_input.user_input}")
            logger.debug(f"Final response: {output}")
            yield f"data: {json.dumps({'type': 'message', 'content': output})}\n\n"

    logger.info("Message stream complete. Sending [DONE] signal.")
    yield "data: [DONE]\n\n"


def _sse_response_example() -> dict[int, Any]:
    return {
        status.HTTP_200_OK: {
            "description": "Server Sent Event Response",
            "content": {
                "text/event-stream": {
                    "example": "data: {'type': 'token', 'content': 'Hello'}\n\ndata: {'type': 'token', 'content': ' World'}\n\ndata: [DONE]\n\n",
                    "schema": {"type": "string"},
                }
            },
        }
    }


@app.post("/dev-stream", response_class=StreamingResponse, responses=_sse_response_example())
async def stream_agent(user_input: UserInput) -> StreamingResponse:
    """
    Stream the agent's response to a user input, including intermediate messages and tokens.

    Use thread_id to persist and continue a multi-turn conversation. run_id kwarg
    is also attached to all messages for recording feedback.
    """
    return StreamingResponse(message_generator(user_input), media_type="text/event-stream")


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    uvicorn.run(app, host=str(os.getenv("HOST")), port=int(os.getenv("PORT")))

