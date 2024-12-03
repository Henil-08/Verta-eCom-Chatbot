from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from components.state import MultiAgentState

def route_question(state):
    source = state['question_type']
    if source.datasource == "Metadata":
        return "Metadata"
    elif source.datasource == "Review-Vectorstore":
        return "Review-Vectorstore"
    elif source.datasource == "FINISH":
        return "FINISH"


def final_llm_node(state: MultiAgentState, prompt, model):
    question = state["question"]
    documents = state["documents"]

    if 'gpt' in model: 
        llm = ChatOpenAI(model_name=model)
    else:
        llm = ChatGroq(model_name=model)

    system_prompt = (
        f"{prompt}"
        "{context}"
    )

    qa_prompt = ChatPromptTemplate.from_messages(
                    [
                        ("system", system_prompt),
                        ("human", "{input}")
                    ]
                )
    question_answer_chain = qa_prompt | llm

    generation = question_answer_chain.invoke({"context": documents, "input": question})

    return {"documents": documents, "question": question, "answer": generation}


def followup_node(state: MultiAgentState, prompt, model):
    documents = state['documents']
    question = state['question']
    answer = state['answer']
    
    if 'gpt' in model: 
        llm = ChatOpenAI(model_name=model)
    else:
        llm = ChatGroq(model_name=model)

    system_prompt = (
        prompt
    )       
    follow_prompt = ChatPromptTemplate.from_messages(
                    [
                        ("system", system_prompt),
                    ]
                )
    
    followup_chain = follow_prompt | llm
    followup = followup_chain.invoke({'question': question, 'answer': answer.content, 'context': documents[-2:]}) # just consider last two document list 
    followup_questions = followup.content.split('\n')
    
    return {"question": question, "answer": answer, "documents": documents, 'followup_questions': followup_questions}       