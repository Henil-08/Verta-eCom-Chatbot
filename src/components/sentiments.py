from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

from logger import logger


def prob_sentiment_model(model_name, prompt):
    if 'gpt' in model_name:
        llm = ChatOpenAI(model_name=model_name)
    else:
        try:
            llm = ChatGroq(model_name=model_name)
        except:
            llm = ChatGroq(model_name="llama-3.1-8b-instant")
            logger.warning(f"Incorrect Sentiment Model name {model_name}, using 'llama-3.1-8b-instant'")

    system_prompt = (
        prompt
    )
    sentiment_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
        ]
    )
    parser = StrOutputParser()
    model = sentiment_prompt | llm | parser

    return model


def sentiment_model(model_name, prompt):
    if 'gpt' in model_name:
        llm = ChatOpenAI(model_name=model_name)
    else:
        try:
            llm = ChatGroq(model_name=model_name)
        except:
            llm = ChatGroq(model_name="llama-3.1-8b-instant")
            logger.warning(f"Incorrect Sentiment Model name {model_name}, using 'llama-3.1-8b-instant'")

    system_prompt = (
        prompt
    )
    sentiment_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
        ]
    )
    parser = StrOutputParser()
    model = sentiment_prompt | llm | parser

    return model