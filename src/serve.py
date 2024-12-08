import json
import os
import uuid
from collections.abc import AsyncGenerator
from datetime import datetime
from typing import Any

import uvicorn
from dotenv import load_dotenv
from fastapi import Body, Depends, FastAPI, HTTPException, Query, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from langfuse.callback import CallbackHandler

from logger import logger
from config.configuration import ConfigurationManager
from pipeline.generation import Generate
from pipeline.stage_01_prepare_base_model import PrepareBaseTrainingPipeline
from pydantic_models.models import Payload, scoreTrace

load_dotenv()

# load the API Keys
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["LANGFUSE_PUBLIC_KEY"] = os.getenv("LANGFUSE_PUBLIC_KEY")
os.environ["LANGFUSE_SECRET_KEY"] = os.getenv("LANGFUSE_SECRET_KEY")
os.environ["LANGFUSE_HOST"] = os.getenv("LANGFUSE_HOST")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["MLFLOW_TRACKING_URI"] = os.getenv("MLFLOW_TRACKING_URI")
os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("MLFLOW_TRACKING_USERNAME")
os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("MLFLOW_TRACKING_PASSWORD")
VERTA_API_ACCESS_TOKEN = os.environ["VERTA_API_ACCESS_TOKEN"]

app = FastAPI()

# Use HTTPBearer as the security scheme
bearer_scheme = HTTPBearer()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def verify_token(credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme)):
    token = credentials.credentials
    if token != VERTA_API_ACCESS_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return token


class ClientApp:
    def __init__(self):
        config = ConfigurationManager()
        prepare_base_model_config = config.get_prepare_base_model_config()
        self.generate = Generate(config=prepare_base_model_config)

        prepare_base = PrepareBaseTrainingPipeline()
        self.app = prepare_base.graph()


clapp = ClientApp()


@app.middleware("http")
async def log_requests(request: Request, call_next):
    # Start time
    start_time = datetime.now()
    response = await call_next(request)

    # End time
    process_time = (datetime.now() - start_time).total_seconds()

    log_data = {
        "timestamp": datetime.now().isoformat(),
        "method": request.method,
        "url": str(request.url),
        "status_code": response.status_code,
        "process_time": process_time,
        "client_ip": request.client.host,
    }

    # Send log to Google Cloud Logging or stdout for Cloud Run to capture
    logger.info(json.dumps(log_data))
    return response


# Define a background task to delete cache files older than 1 hour
# async def clear_outdated_cache():
#     while True:
#         current_time = datetime.utcnow()
#         for cache_key, creation_time in list(cache_creation_times.items()):
#             if (current_time - creation_time) > timedelta(hour=1):
#                 try:
#                     # Remove cache files
#                     if os.path.exists(f"{faiss_dir}/{cache_key}") and os.path.isdir(f"{faiss_dir}/{cache_key}"):
#                         shutil.rmtree(f"{faiss_dir}/{cache_key}")
#                     if os.path.exists(f"{meta_dir}/{cache_key}.csv") and os.path.isfile(f"{meta_dir}/{cache_key}.csv"):
#                         os.remove(f"{meta_dir}/{cache_key}.csv")

#                     # Remove from cache tracking
#                     vector_store_cache.remove(cache_key)
#                     del cache_creation_times[cache_key]

#                     logger.info(f"Cache automatically cleared for {cache_key}")
#                 except Exception as e:
#                     logger.error(f"Error clearing cache for {cache_key}: {e}")

#         # Wait an hour before the next check
#         await asyncio.sleep(3600)


# # Use lifespan to manage startup and shutdown events
# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     # Startup tasks
#     cache_task = asyncio.create_task(clear_outdated_cache())
#     try:
#         yield
#     finally:
#         cache_task.cancel()
#         await cache_task  # Wait for the task to be cancelled

# app.router.lifespan_context = lifespan


@app.get("/")
async def health():
    return {"status": "🤙"} 


@app.get("/initialize")
async def initialize(
    token: str = Depends(verify_token),
    asin: str = Query(...),
    user_id: int = Query(...),
):
    logger.info(
        f"Received request to initialize retriever for ASIN: {asin} and User ID: {user_id}"
    )
    await clapp.generate.initialize(asin, user_id)
    return JSONResponse(
        content={"status": "retriever initialized", "asin": asin, "user_id": user_id},
        status_code=200,
    )


@app.post("/score")
async def annotate(
    token: str = Depends(verify_token),
    score: scoreTrace = Body(..., description="json for scoring feedback"),
):
    try:
        response = await clapp.generate.score_feedback(score)
        return response
    except Exception as e:
        logger.error(f"Error Scoring Trace: {score.run_id} - {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/dev-invoke")
async def invoke(
    token: str = Depends(verify_token),
    payload: Payload = Body(..., description="json for user query"),
):
    logger.info(
        f"Received request to invoke agent for ASIN: {payload.parent_asin} and User ID: {payload.user_id}"
    )

    asin = payload.parent_asin
    user_id = payload.user_id
    cache_key = f"{user_id}-{asin}"

    # Ensure paths exist
    retriever_path, metadata_path = await clapp.generate.initialize(
        asin, user_id, returnPath=True
    )

    if not os.path.exists(retriever_path):
        logger.error(
            f"Retriever not initialized for ASIN: {payload.parent_asin} and User ID: {payload.user_id}"
        )
        return JSONResponse(
            content={"status": "Retriever not initialized"}, status_code=400
        )
    if not os.path.exists(metadata_path):
        logger.error(
            f"Meta-Data not initialized for ASIN: {payload.parent_asin} and User ID: {payload.user_id}"
        )
        return JSONResponse(
            content={"status": "Meta-Data not initialized"}, status_code=400
        )

    agent = clapp.app
    
    if payload.log_langfuse:
        run_id = str(uuid.uuid4())
        langfuse_handler = CallbackHandler(
            user_id=f"{payload.user_id}", session_id=f"{cache_key}"
        )
        config = {"callbacks": [langfuse_handler], "run_id": run_id}
    try:
        response = await agent.ainvoke(
            {
                "question": payload.query,
                "meta_data": str(metadata_path),
                "retriever": str(retriever_path),
            },
            config=config,
        )

        logger.info(f"Agent response generated for User ID: {payload.user_id}")
        output = {
            "run_id": run_id,
            "question": response["question"],
            "answer": response["answer"].content,
            "followup_questions": response["followup_questions"],
        }
        logger.debug(f"Final response: {output}")
        return output
    except Exception as e:
        logger.error(f"Error invoking agent for User ID: {payload.user_id} - {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def message_generator(
    payload: Payload, stream_tokens=True
) -> AsyncGenerator[str, None]:
    """
    Generate a stream of messages from the agent.

    This is the workhorse method for the /stream endpoint.
    """
    asin = payload.parent_asin
    user_id = payload.user_id
    cache_key = f"{user_id}-{asin}"
    logger.info(f"Initializing message generator for cache key: {cache_key}")

    # Ensure paths exist
    retriever_path, metadata_path = await clapp.generate.initialize(
        asin, user_id, returnPath=True
    )

    if not os.path.exists(retriever_path):
        logger.error(
            f"Retriever not initialized for ASIN: {payload.parent_asin} and User ID: {payload.user_id}"
        )
        yield JSONResponse(
            content={"status": "Retriever not initialized"}, status_code=400
        )
    if not os.path.exists(metadata_path):
        logger.error(
            f"Meta-Data not initialized for ASIN: {payload.parent_asin} and User ID: {payload.user_id}"
        )
        yield JSONResponse(
            content={"status": "Meta-Data not initialized"}, status_code=400
        )

    agent = clapp.app
    if payload.log_langfuse:
        run_id = str(uuid.uuid4())
        langfuse_handler = CallbackHandler(
            user_id=f"{payload.user_id}", session_id=f"{cache_key}"
        )
        config = {"callbacks": [langfuse_handler], "run_id": run_id}
    if payload.stream_tokens == 0:
        stream_tokens = False

    logger.info("Starting event stream processing for agent.")

    # Process streamed events from the graph and yield messages over the SSE stream.
    async for event in agent.astream_events(
        {
            "question": payload.query,
            "meta_data": str(metadata_path),
            "retriever": str(retriever_path),
        },
        version="v2",
        config=config,
    ):
        if not event:
            logger.warning("Received empty event in stream.")
            continue

        # Yield tokens streamed from LLMs.
        if (
            event["event"] == "on_chat_model_stream"
            and stream_tokens == True
            and any(t.startswith("seq:step:2") for t in event.get("tags", []))
            and event["metadata"]["langgraph_node"] == "generate"
        ):
            content = event["data"]["chunk"].content
            if content:
                logger.debug(f"Streaming token: {content}")
                yield f"data: {json.dumps({'type': 'token', 'content': content})}\n\n"
            continue

        # Yield messages written to the graph state after node execution finishes.
        if (event["event"] == "on_chain_end") and (
            (any(t.startswith("seq:step:2") for t in event.get("tags", [])))
            and (
                (event["metadata"]["langgraph_node"] == "final")
                and (event["metadata"]["langgraph_triggers"] == ["generate"])
            )
        ):
            answer = event["data"]["output"]["answer"].content
            followup_questions = event["data"]["output"]["followup_questions"]
            output = {
                "run_id": run_id,
                "question": payload.query,
                "answer": answer,
                "followup_questions": followup_questions,
            }
            logger.info(f"Yielding final response for User ID: {payload.user_id}")
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


@app.post(
    "/dev-stream", response_class=StreamingResponse, responses=_sse_response_example()
)
async def stream_agent(
    token: str = Depends(verify_token),
    payload: Payload = Body(..., description="json for user query"),
) -> StreamingResponse:
    """
    Stream the agent's response to a user input, including intermediate messages and tokens.

    Use thread_id to persist and continue a multi-turn conversation. run_id kwarg
    is also attached to all messages for recording feedback.
    """
    return StreamingResponse(message_generator(payload), media_type="text/event-stream")


if __name__ == "__main__":
    uvicorn.run(app, host=str(os.getenv("HOST")), port=int(os.getenv("PORT")))
