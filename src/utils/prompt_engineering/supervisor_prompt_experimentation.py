# supervisor_prompt_experimentation.py
import os
from dotenv import load_dotenv
import pandas as pd
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from state import MultiAgentState, RouteQuery
from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge
from helpers import compute_cosine_similarity
from constants import MEMBERS, OPTIONS
from langchain_openai import ChatOpenAI


# Load environment variables
load_dotenv('.env')
print(os.getenv("OPENAI_API_KEY")) 
print(os.getenv("GROQ_API_KEY")) 
os.environ['OPENAI_API_KEY']=os.getenv("OPENAI_API_KEY")
os.environ['GROQ_API_KEY']=os.getenv("GROQ_API_KEY")
def supervisor_prompt_experimentation(state, test_cases):
    """
    Performs prompt experimentation for the Supervisor Agent, evaluates each response,
    and selects the best prompt based on evaluation metrics.

    Args:
        state (MultiAgentState): The current state containing metadata.
        expected_response (str): The expected response for evaluation.

    Returns:
        best_prompt (dict): The best prompt and its evaluation metrics.
    """
    supervisor_llm = ChatOpenAI(model_name="gpt-4o-mini")

    # Original and New Prompts
    prompts = [
        ('Prompt 1', '''
            You are an efficient supervisor responsible for overseeing a conversation between the following agents: {members}. 

            If you got response from the Agent (response given below as "Generated Answer from the Agents:"), respond with 'FINISH' to move on to next step. 
            
            Based on the user's request, decide which agent should respond next. Each agent will complete a task and return their result. 
            
            There are two agents working alongside you:
                - Metadata: This agent has all metadata information about that product. 
                - Review-Vectorstore: This is a FAISS Vectorstore db containing documents related to all the user reviews for that product.
            
            If you got unsatisfied response from the Agents (Agent Throwing Errors like: "Metadata: Unable to generate result") ONLY THEN Call an Agent a **MAXIMUM of TWO TIMES** before responding with 'FINISH'.
            Once sufficient information is obtained from the Agents, respond with 'FINISH', after which Alpha, the final assistant, will provide the concluding guidance to the user.
            If the query is generic (Hello, How are you, etc) then route it to Alpha and respond with 'FINISH.' 

            If you got satisfactory response from the Agent (response given above), respond with 'FINISH' to move on to next step. 
        '''),
        ('Prompt 2', '''
            You are an efficient supervisor responsible for overseeing a conversation between agents: {members}. 

            Your role is to route each query to the most suitable agent based on the query’s content. Evaluate responses (provided below as "Generated Answer from the Agents:") to ensure completeness.

            Instructions:
            1. Choose the relevant agent from {members} to handle each query.
            2. If the provided information fully addresses the query, respond with 'FINISH' to proceed.
            3. When simple or general queries arise, route them directly to Alpha for an answer and reply with 'FINISH' after.

            Agents Available:
            - Metadata: Manages structured product information.
            - Review-Vectorstore: Provides insights from user reviews.

        '''),
        ('Prompt 3', '''
            You are the supervising agent for conversations involving {members}. Your task is to assign queries to the best agent and ensure users receive complete responses.

            Instructions:
            1. Review each "Generated Answer from the Agents:" and determine if it fulfills the query.
            2. Direct queries to the correct agent in {members} based on the type of information requested.
            3. If you believe the answer is sufficient, conclude the process by responding with 'FINISH.'

            If a straightforward query is detected, pass it directly to Alpha and end with 'FINISH.'
        '''),
        ('Prompt 4', '''
            Act as the supervisor overseeing {members}. Your responsibility is to manage agent responses by choosing the best agent for each query, ensuring responses are accurate and complete.

            Guidelines:
            1. Route queries to one of the agents in {members} based on the query's nature and information requested.
            2. Review responses under "Generated Answer from the Agents:" and determine if they are complete.
            3. When information is fully provided, respond with 'FINISH.'

            For general questions, delegate them to Alpha, and then respond with 'FINISH' to indicate completion.

        '''),
        ('Prompt 5', '''
            As a supervisor for {members}, your role is to route each user query to the most relevant agent and ensure responses are complete and accurate.

            Instructions:
            1. Select the best agent from {members} based on the specific information required.
            2. After each response ("Generated Answer from the Agents:"), assess if it satisfies the query.
            3. Respond with 'FINISH' if the answer is complete or if a general question has been routed to Alpha.

            Agent Overview:
            - Metadata: Handles product data and specifications.
            - Review-Vectorstore: Manages user reviews and insights.
        '''),
        ('Prompt 6', '''
            You are the supervisor responsible for guiding queries through {members}. For each query, decide which agent should respond next, aiming for accurate and complete information.

            Guidelines:
            1. Choose an agent from {members} based on the type of response required.
            2. Review each "Generated Answer from the Agents:" to determine if the information provided is sufficient.
            3. If the response is satisfactory, reply with 'FINISH' to proceed.

            When encountering straightforward questions, route them to Alpha and conclude with 'FINISH.'

         ''')
    ]

    results = []
    rouge = Rouge()

    for prompt_name, prompt_template in prompts:
        success_count = 0
        failure_count = 0
        for test_case in test_cases:
            state['question'] = test_case["query"]
            prompt = prompt_template.format(members=", ".join(OPTIONS))

            # Build the prompt
            supervisor_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", prompt),
                    ("human", state['question']),
                    ("system", "Generated Answer from the Agents: " + state['documents'][-1] if state['documents'] else "")
                ]
            )
            parser = StrOutputParser()
            supervisor_chain = supervisor_prompt | supervisor_llm.with_structured_output(RouteQuery)

            try:
                # Generate response
                supervisor_result = supervisor_chain.invoke({'input': ''})
                response = supervisor_result.datasource
            except Exception as error:
                print(f"Error with {prompt_name}: {error}")
                response = "Error generating response."

            # Check if response matches the expected single-word response
            if response.lower() == test_case["expected_response"].lower():
                success_count += 1
            else:
                print(f"Query: {test_case['query']}")
                print(f"Expected Response: {test_case['expected_response']}")
                print(f"Generated Response: {response}")
                failure_count += 1

        # Calculate success rate
        success_rate = success_count / len(test_cases) * 100
        failure_rate = failure_count / len(test_cases) * 100

        results.append({
            'prompt_name': prompt_name,
            'prompt': prompt,
            'success_rate': success_rate,
            'failure_rate': failure_rate
        })

        print(f"\n{prompt_name} Evaluation:")
        print(f"Success Rate: {success_rate:.2f}%")
        print(f"Failure Rate: {failure_rate:.2f}%\n")

    # Determine the best prompt based on Success Rate
    best_prompt = max(results, key=lambda x: x['success_rate'])

    print(f"\nBest Prompt: {best_prompt['prompt_name']} with Success Rate: {best_prompt['success_rate']:.2f}%")
    return best_prompt

if __name__ == "__main__":
    # Define sample expected response for testing
    test_cases = [
        {"query": "Can you tell me the product’s dimensions?", "expected_response": OPTIONS[1]},
        {"query": "What do other customers think about this product?", "expected_response": OPTIONS[2]},
        {"query": "Hello, how are you today?", "expected_response": OPTIONS[0]},
        {"query": "I need to know more about this product.", "expected_response": OPTIONS[1]},
        {"query": "What are the common pros and cons mentioned by users?", "expected_response": OPTIONS[2]},
        {"query": "Does this product come in multiple colors?", "expected_response": OPTIONS[1]},
        {"query": "Are there any complaints about this product’s durability?", "expected_response": OPTIONS[2]},
        {"query": "What’s the product’s warranty period?", "expected_response": OPTIONS[1]},
        {"query": "Just browsing—what’s new?", "expected_response": OPTIONS[0]},
        {"query": "Why should I choose this product over others?", "expected_response": OPTIONS[0]},
        {"query": "Is this product reliable according to data?", "expected_response": OPTIONS[2]},
        {"query": "Can you check if there are any discounts for this product?", "expected_response": OPTIONS[1]},
        {"query": "Please provide specifications.", "expected_response": OPTIONS[1]},
        {"query": "Show me the best features from user reviews.", "expected_response": OPTIONS[2]},
        {"query": "Only show me user reviews.", "expected_response": OPTIONS[2]},
        {"query": "Can you summarize the main details?", "expected_response": OPTIONS[1]},
        {"query": "I didn’t get a response to my last question.", "expected_response": OPTIONS[0]},
        {"query": "What’s the most common feedback about this product?", "expected_response": OPTIONS[2]},
        {"query": "Is this product on sale?", "expected_response": OPTIONS[1]},
        {"query": "How is the product rated by users?", "expected_response": OPTIONS[2]}
    ]

    # Define sample state with necessary elements
    state = MultiAgentState(
        question="What are the specifications of this product?",
        question_type="Product Inquiry",
        answer="",
        documents=["Response from Review-Vectorstore: Unable to provide specifications."],
        meta_data=None,
        retriever=None,
        followup_questions=None
    )

    # Run supervisor prompt experimentation
    best_prompt = supervisor_prompt_experimentation(state, test_cases)