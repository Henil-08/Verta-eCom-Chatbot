from pandas import DataFrame
from operator import add
from pydantic import BaseModel
from typing import Annotated, Literal, List
from typing_extensions import TypedDict

from langchain_core.documents import Document
from langchain_core.vectorstores.base import VectorStoreRetriever

from constants import OPTIONS

class MultiAgentState(TypedDict):
    question: str
    meta_data: DataFrame
    retriever: VectorStoreRetriever
    meta_summary: Document
    question_type: str
    documents: Annotated[List[str], add]
    answer: str 
    followup_questions: list[str]

class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""
    datasource: Literal[*OPTIONS] # type: ignore
