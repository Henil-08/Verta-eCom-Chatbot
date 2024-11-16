from pydantic import BaseModel

class Payload(BaseModel):
    query: str
    parent_asin: str
    user_id: str
    log_langfuse: bool
    stream_tokens: bool

class scoreTrace(BaseModel):
    run_id: str
    user_id: str
    parent_asin: str
    value: bool