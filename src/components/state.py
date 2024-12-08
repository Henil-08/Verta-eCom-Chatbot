from pandas import DataFrame
from operator import add
from pydantic import BaseModel
from typing import Annotated, Literal, List
from typing_extensions import TypedDict

from langchain_core.vectorstores.base import VectorStoreRetriever

from constants import OPTIONS

class MultiAgentState(TypedDict):
    question: str
    question_type: str
    answer: str 
    documents: Annotated[List[str], add]
    meta_data: DataFrame
    retriever: VectorStoreRetriever
    followup_questions: list[str]

class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""
    datasource: Literal[*OPTIONS] # type: ignore
