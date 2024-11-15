# final_llm_prompt_experimentation.py
import os
from dotenv import load_dotenv
import pandas as pd
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEmbeddings
from state import MultiAgentState
from langchain.schema import Document
from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge
from helpers import compute_cosine_similarity
from sqlalchemy import create_engine, text

from sqlalchemy import create_engine

# Load environment variables
load_dotenv()
DB_URI = os.getenv("DB_URI")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
os.environ['GROQ_API_KEY'] = GROQ_API_KEY


# Function to retrieve metadata from the database
def fetch_metadata_from_db(asin):
    engine = create_engine(DB_URI)
    with engine.connect() as connection:
        meta_query = text(f"""
            SELECT parent_asin, main_category, title, average_rating, rating_number, features, description, price, store, categories, details
            FROM metadata 
            WHERE parent_asin = :asin
        """)
        meta_result = connection.execute(meta_query, {'asin': asin})
        meta_df = pd.DataFrame(meta_result.fetchall(), columns=meta_result.keys())
    return meta_df

# Function to retrieve review data from the database and set up a retriever
def setup_review_retriever(asin):
    review_result = [
        (
            "B072K6TLJX",             # parent_asin
            "B072K6TLJX",             # asin
            34,                       # helpful_vote
            "2021-08-14 13:45:00",    # timestamp
            True,                     # verified_purchase
            "Great product for LOTR fans!",  # title
            "As a Lord of the Rings fan, I absolutely love this FunKo POP! Frodo figurine. The details are fantastic, and it looks great on my shelf. Highly recommend for any collector."
        ),
        (
            "B072K6TLJX",
            "B072K6TLJX",
            18,
            "2021-09-10 09:30:00",
            False,
            "Good but slightly overpriced",
            "The Frodo FunKo POP! figure is well-made and detailed, but I found it a bit overpriced for what you get. Otherwise, it's a solid addition to any collection."
        ),
        (
            "B072K6TLJX",
            "B072K6TLJX",
            22,
            "2021-07-20 14:18:00",
            True,
            "Perfect for fans and collectors",
            "This Frodo figure is so cute and has great detail! It's a must-have for any LOTR fan. Love the packaging as well, arrived in perfect condition."
        ),
        (
            "B072K6TLJX",
            "B072K6TLJX",
            5,
            "2021-06-01 10:22:00",
            True,
            "Not as detailed as expected",
            "I was a bit disappointed with the detail on this POP! figure. It’s still good quality, but I expected a bit more for the price. It’s still cute, though!"
        ),
        (
            "B072K6TLJX",
            "B072K6TLJX",
            45,
            "2021-05-25 16:45:00",
            True,
            "Frodo Funko is a win!",
            "I have a whole collection of Funko POPs, and this Frodo Baggins one is one of my favorites. The colors are vibrant, and it’s sturdy and well-made."
        ),
        (
            "B072K6TLJX",
            "B072K6TLJX",
            12,
            "2021-05-10 08:50:00",
            False,
            "Nice collectible, okay quality",
            "The Frodo figure is nice to look at, but I feel the quality isn’t as great as other Funko POP! items I own. It feels a bit light and fragile."
        ),
        (
            "B072K6TLJX",
            "B072K6TLJX",
            30,
            "2021-04-30 15:20:00",
            True,
            "My son loves it!",
            "Bought this for my son who’s a huge LOTR fan, and he was thrilled. Great addition to his collection, and it arrived quickly."
        ),
        (
            "B072K6TLJX",
            "B072K6TLJX",
            27,
            "2021-04-22 17:35:00",
            True,
            "Fantastic detail and finish",
            "The details on this Frodo figure are excellent, especially the texture on the cloak and hair. Worth every penny for collectors."
        ),
        (
            "B072K6TLJX",
            "B072K6TLJX",
            19,
            "2021-04-18 10:10:00",
            False,
            "Smaller than expected but great quality",
            "The figure is smaller than I expected, but the quality is great, and it fits well on my display shelf with other LOTR memorabilia."
        ),
        (
            "B072K6TLJX",
            "B072K6TLJX",
            40,
            "2021-03-30 18:55:00",
            True,
            "A must-have for LOTR fans",
            "This is a fantastic collectible for any Lord of the Rings fan. The quality is top-notch, and the paint job is flawless. Highly recommend!"
        ),
    ]

    columns = ["parent_asin", "asin", "helpful_vote", "timestamp", "verified_purchase", "title", "text"]
    review_df = pd.DataFrame(review_result, columns=columns)

    # Load the reviews into the FAISS vector store
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    review_documents = [Document(page_content=row['text'], metadata=row.to_dict()) for index, row in review_df.iterrows()]
    vectordb = FAISS.from_documents(review_documents, embedding=embeddings)
    retriever = vectordb.as_retriever()
    
    return retriever

# Experimentation function for final LLM prompts
def final_llm_prompt_experimentation(state, questions_with_answers):
    """
    Performs prompt experimentation for the final LLM, evaluates each response,
    and selects the best prompt based on evaluation metrics.
    
    Args:
        state (MultiAgentState): The current state containing metadata and question.
        expected_answer (str): The expected answer for evaluation.

    Returns:
        best_prompt (dict): The best prompt and its evaluation metrics.
    """
    llm = ChatGroq(model_name="llama-3.1-70b-versatile")

    # List of prompt variations for experimentation
    prompts = [
        ('Prompt 1', f'''
        You are Alpha, a highly knowledgeable and efficient chatbot assistant designed to help users with questions related to products.
        Your primary role is to assist users by providing concise, accurate, and insightful responses based on the product information and reviews available to you.
        If you don’t have the necessary information to answer the question, simply say that you don’t know.

        There are two agents working alongside you:
        - Metadata: This agent provides answers related to a product. It has all the information about that product.
        - Review-Vectorstore: This is a FAISS Vectorstore db containing documents related to all the user reviews for one product.
        
        When a User (Shopper) comes to you for help, the question might have first been routed through either the Metadata or the Review-Vectorstore. 

        Your primary objective is to offer clear, concise, and helpful advice to the teacher, ensuring that they receive the most accurate and useful information to support their shopping needs.

        Instructions:
        - Analyze the product information and/or reviews provided.
        - Provide brief, clear, and helpful answers to user queries about the product.
        - Focus on delivering concise and actionable insights to help users make informed decisions.

        The responses from those agents are available to you, and if their answers were incomplete or unsatisfactory, you will find this reflected in the context field. 
        Your job is to analyze their responses, determine if they are adequate, and provide additional guidance or clarification where needed.
        Below is the context from one of the agents:
        '''
        "{context}" 
        ),
        ('Prompt 2', f'''
            You are Alpha, a dedicated and insightful product assistant designed to provide users with accurate and helpful information about products.
            Your main role is to assist users by leveraging both product metadata and user reviews to deliver clear, brief, and useful responses.

            Agents assisting you:
            - Metadata: Provides specific details about the product, including features, specifications, and pricing.
            - Review-Vectorstore: A database of user reviews that offers insights into customer experiences and satisfaction.

            Instructions:
            - Review the product information and/or user feedback provided.
            - Respond to the user’s questions with clear, concise, and relevant answers.
            - Focus on providing actionable information that supports users in making informed purchasing decisions.

            When information from the agents is incomplete, you will find it reflected in the provided context. Use your understanding of the product to offer guidance or clarification where necessary. Here is the context from the agents:
            '''
            "{context}"
        ),
        ('Prompt 3', f'''
            You are Alpha, a highly skilled and informative assistant focused on helping users understand product details and reviews.
            Your task is to support users by analyzing product metadata and user reviews to provide quick, accurate answers to their questions.

            Supporting Agents:
            - Metadata: Holds detailed information about the product, including attributes, pricing, and specifications.
            - Review-Vectorstore: Contains a database of user reviews for the product, capturing customer feedback and experiences.

            Instructions:
            - Use the provided product data and reviews to answer user questions clearly and accurately.
            - Ensure each response is concise, helpful, and relevant to the user’s needs.
            - Offer actionable advice that aids the user in making a well-informed purchase decision.

            The responses from these agents are provided as context. If you notice gaps or incomplete answers, analyze and clarify accordingly. Here is the context:
            '''
            "{context}"
        ),
        ('Prompt 4', f'''
            You are Alpha, an intelligent product assistant here to provide users with straightforward answers to questions about products.
            Using the product metadata and user reviews, your goal is to deliver accurate and concise responses that help users with their purchasing decisions.

            Assisting Agents:
            - Metadata: Supplies product-specific details, such as features, specifications, and pricing.
            - Review-Vectorstore: A database of user reviews, offering real-world insights into customer experiences.

            Guidelines:
            - Examine the product information and user feedback to address user questions clearly and effectively.
            - Aim for responses that are concise, accurate, and directly relevant to the user’s needs.
            - Ensure users get actionable information that helps them assess the product thoroughly.

            If there is incomplete or partial information, this will be noted in the context. Evaluate the context provided by agents and offer additional clarification as needed. Here’s the context:
            '''
            "{context}"
        ),
        ('Prompt 5', f'''
            You are Alpha, a knowledgeable and reliable product assistant tasked with helping users make informed purchasing decisions.
            Your responses should draw from both product metadata and user reviews to ensure accuracy and relevance.

            Agents Supporting You:
            - Metadata: Contains detailed product attributes, including key features, specifications, and price.
            - Review-Vectorstore: Houses a collection of user reviews that provide insights into the product’s real-world performance and user satisfaction.

            Guidelines:
            - Use the product details and reviews to provide clear and helpful answers to user questions.
            - Your answers should be concise, relevant, and focused on the user’s needs.
            - Strive to give users actionable insights that support informed decision-making.

            When answers from agents are insufficient or lack detail, it will be indicated in the context. Review the context provided and clarify as needed. Here’s the context from the agents:
            '''
            "{context}"
        )
    ]

    results = []
    rouge = Rouge()
    for question, expected_answer in questions_with_answers:
        documents = state['retriever'].get_relevant_documents(question)
        for prompt_name, prompt_template in prompts:
            # Format the prompt with metadata
            # prompt = prompt_template.format(question=question, documents=documents)

            # Build the prompt
            qa_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", prompt_template),
                    ("human", "{input}")
                ]
            )
            parser = StrOutputParser()
            llm_chain = qa_prompt | llm | parser

            try:
                # Generate response
                response = llm_chain.invoke({"context": documents, "input": question})
            except Exception as error:
                print(f"Error with {prompt_name}: {error}")
                response = "Error generating response."

            # Evaluate the response
            bleu_score = sentence_bleu([expected_answer.split()], response.split())
            rouge_scores = rouge.get_scores(response, expected_answer, avg=True)
            rouge_l_f1 = rouge_scores['rouge-l']['f']
            cosine_similarity = compute_cosine_similarity(response, expected_answer)

            # Save the results
            results.append({
                'question': question,
                'expected_answer': expected_answer,
                'prompt_name': prompt_name,
                'response': response,
                'bleu_score': bleu_score,
                'rouge_l_f1': rouge_l_f1,
                'cosine_similarity': cosine_similarity
            })
            print(f"\n{prompt_name} Evaluation:")
            print(f"BLEU Score: {bleu_score:.4f}")
            print(f"ROUGE-L F1 Score: {rouge_l_f1:.4f}")
            print(f"Cosine Similarity: {cosine_similarity:.4f}")
    df = pd.DataFrame(results)

    # Group by question and select the row with the max cosine similarity
    best_per_question = df.loc[df.groupby('question')['cosine_similarity'].idxmax()]

    # Count the frequency of each selected prompt
    prompt_counts = best_per_question['prompt_name'].value_counts()

    # Identify the most frequently selected prompt
    most_frequent_prompt = prompt_counts.idxmax()
    print(f"Most frequent prompt: {most_frequent_prompt}")

    return most_frequent_prompt

if __name__ == "__main__":
    # Define the ASIN for the product and fetch metadata and review data
    asin = "B072K6TLJX"
    data = {
        'main_category': ['Toys & Games'],
        'title': ['FunKo POP! Movies Lord of the Rings Frodo Baggins Vinyl Figure'],
        'average_rating': [4.8],
        'rating_number': [1234],
        'features': ['Stylized collectable stands 3 ¾ inches tall, perfect for any Lord of the Rings fan!'],
        'description': ['From Lord of the Rings, Frodo Baggins, as a stylized POP vinyl from Funko! Figure stands 3 3/4 inches and comes in a window display box.'],
        'price': ['$10.99'],
        'store': ['FunKo Store'],
        'categories': ['Collectibles, Vinyl Figures'],
        'details': ['Product Dimensions: 2.5 x 2.5 x 3.75 inches; Item Weight: 4 ounces; Manufacturer: FunKo']
    }
    meta_df = pd.DataFrame(data)
    retriever = setup_review_retriever(asin)

    questions_with_answers = [
         (
        "Does this product come in different colors?",
        "There is no mention of this product being available in different colors in the reviews or metadata."
    ),
    (
        "Is this product durable?",
        "Many reviews describe the product as durable and well-made, though a few users expected higher quality in details."
    ),
    (
        "Is this product lightweight?",
        "The product is lightweight and compact, as highlighted by several reviewers."
    ),
    (
        "Can kids use it?",
        "The reviews and metadata do not specify whether the product is suitable for children, but it is generally described as a collectible rather than a toy."    ),
    (
        "Does it come in different materials?",
        "The metadata and reviews do not mention if this product is available in different materials. It is made of vinyl as typical for FunKo POP! figures."
    ),
    (
        "What is the average rating of this product?",
        "The product has an average rating of 4.8 out of 5 based on 1,234 customer reviews."
    ),
    (
        "Is it worth the price?",
        "Most reviewers consider the product worth the price due to its quality and collectible value, though a few felt it was slightly overpriced."
    ),
    (
        "Does it come with a display box?",
        "Yes, the product comes in a window display box, as mentioned in the metadata and reviews."
    ),
    (
        "What do people like most about this product?",
        "Customers love the detailed craftsmanship, compact size, and nostalgic appeal of the product."
    ),
    (
        "Are there any issues with this product?",
        "Some reviewers noted that the product could have better detailing, and a few mentioned it being slightly overpriced."
    )
    ]
    # Prepare state
    state = MultiAgentState(
        question="",
        question_type="",
        answer="",
        meta_data=meta_df,
        retriever=retriever,
        followup_questions=None
    )

    # Define an expected answer for evaluation
    # expected_answer = 'Based on the reviews and metadata, there is no mention of this product being available in different colors.'
    # Run the prompt experimentation
    best_prompt = final_llm_prompt_experimentation(state, questions_with_answers)
