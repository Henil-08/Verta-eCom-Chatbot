import pandas as pd
from langchain_groq import ChatGroq
from langchain.schema import Document
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.output_parsers import StrOutputParser

from constants import MEMBERS, OPTIONS
from components.state import MultiAgentState, RouteQuery


def supervisor_agent(state: MultiAgentState, prompt, model):
    question = state["question"]
    documents = state["documents"]

    system_prompt = (
        prompt
    )

    if 'gpt' in model: 
        llm = ChatOpenAI(model_name=model)
    else:
        llm = ChatGroq(model_name=model)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{question}"),
            ('system', "Generated Answer from the Agents: {document}"),
            (
                "system",
                "Given the conversation above, who should act next?"
                "Or should we FINISH? Select one of: {options}",
            ),
        ]
    ).partial(options=str(OPTIONS), members=", ".join(MEMBERS))

    supervisor_chain = prompt | llm.with_structured_output(RouteQuery)
    question_type = supervisor_chain.invoke({"question": question, 'document': documents})

    return {'question_type' : question_type}


def metadata_node(state: MultiAgentState, prompt, model):

    if 'gpt' in model: 
        llm = ChatOpenAI(model_name=model)
    else:
        llm = ChatGroq(model_name=model)

    meta_df = state['meta_data']

    if isinstance(meta_df, str):
        meta_df = pd.read_csv(meta_df)
    
    modified_details = meta_df['details'].astype(str).str.replace('{', '[')
    
    # Answer question
    meta_system_prompt = ( 
        prompt.format(
            main_category=(meta_df.at[0, 'main_category']),
            title=(meta_df.at[0, 'title']),
            average_rating=(meta_df.at[0, 'average_rating']),
            rating_number=(meta_df.at[0, 'rating_number']),
            features=(meta_df.at[0, 'features']),
            description=(meta_df.at[0, 'description']),
            price=(meta_df.at[0, 'price']),
            store=(meta_df.at[0, 'store']),
            categories=(meta_df.at[0, 'categories']),
            details=(modified_details.at[0]),
        )
    )

    meta_system_prompt = meta_system_prompt.replace('{', '{{').replace('}', '}}')

    meta_qa_prompt = ChatPromptTemplate.from_messages(
                    [
                        ("system", meta_system_prompt),
                    ]
                )
    parser = StrOutputParser()
    meta_chain = meta_qa_prompt | llm | parser

    try:
        # Meta Summary
        meta_results = meta_chain.invoke({'input': ''})
        meta_results = Document(page_content=meta_results, metadata={"source": "Metadata"})
        
    except Exception as error:
        print(error)
        content = "Metadata: Unable to generate result"
        meta_results = Document(page_content=content, metadata={"source": "Metadata"})

    return {'meta_summary': meta_results}


def retrieve(state: MultiAgentState):
    """
    Retrieve documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    question = state["question"]
    retriever = state["retriever"]

    # Load the database
    if isinstance(retriever, str):
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectordb = FAISS.load_local(retriever, embeddings, allow_dangerous_deserialization=True)
        retriever = vectordb.as_retriever()
       
    # Retrieval
    documents = retriever.invoke(question)

    return {"documents": documents}


