import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from utils.prompt_engineering.helpers import compute_cosine_similarity
from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge
from utils.state import MultiAgentState
import pandas as pd
from utils.prompt_engineering.final_llm_prompt_experiment import setup_review_retriever

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
os.environ['GROQ_API_KEY'] = GROQ_API_KEY


def followup_node_prompt_experimentation(state, expected_followup):
    """
    Performs prompt experimentation for the follow-up node, evaluates each response,
    and selects the best prompt based on evaluation metrics.

    Args:
        state (dict): Contains question and additional context.
        expected_followup (str): The expected follow-up question for evaluation.

    Returns:
        best_prompt (dict): The best prompt and its evaluation metrics.
    """
    llm = ChatGroq(model_name="llama-3.1-8b-instant")
    # List of prompt variations for experimentation
    prompts = [
        ('Prompt 1', '''
        Given the following:
        User Question: {question}
        Answer: {answer}
        Context: {context}
        Please generate three possible follow-up questions that the user might ask, each on a new line, without any numbering or bullet points. Do not include any explanations—just list the follow-up questions.
        Format them like this:
        question1\nquestion2\nquestion3
        '''),
        ('Prompt 2', '''
        Using the following information:
        - Initial Question: {question}
        - Current Answer: {answer}
        - Additional Context: {context}
        Generate three user-centric follow-up questions that would logically follow from the given inputs. Each question must appear as a separate line in the format:
        question1\nquestion2\nquestion3
        Only include the questions, nothing else.
        '''),
        ('Prompt 3', '''
        Considering the following:
        - User Question: {question}
        - Answer Provided: {answer}
        - Additional Context: {context}
        Generate three relevant follow-up questions that the user might ask next. Each question should be concise, directly connected to the provided context, and appear on separate lines in this format:
        question1\nquestion2\nquestion3
        No explanations or extra formatting—just the questions.

        '''),
        ('Prompt 4', '''
        Based on the input below:
        User Query: {question}
        Response: {answer}
        Supporting Context: {context}
        Write three follow-up questions that align with the given question and context. Ensure each question appears on a new line and follows this format:
        question1\nquestion2\nquestion3
        Do not include explanations, summaries, or extra formatting—only list the questions.

        '''),
        ('Prompt 5', '''
        Using the following input:
        - User Question: {question}
        - Provided Answer: {answer}
        - Additional Context: {context}
        Craft three possible follow-up questions that the user might ask. The questions should be formatted on separate lines as follows:
        question1\nquestion2\nquestion3
        Do not include any commentary, just the questions in the specified format.
        ''')
    ]

    results = []
    rouge = Rouge()

    for question, answer, expected_followup in expected_followup:
        documents = state['retriever'].get_relevant_documents(question)
        for prompt_name, prompt_template in prompts:
            # Build the prompt
            followup_prompt = ChatPromptTemplate.from_messages(
                    [
                        ("system", prompt_template),
                    ]
                )
            parser = StrOutputParser()
            llm_chain = followup_prompt | llm | parser

            try:
                # Generate response
                response = llm_chain.invoke({'question': question, 'answer': answer, 'context': documents[-2:]})
            except Exception as error:
                print(f"Error with {prompt_name}: {error}")
                response = "Error generating response."

            # Evaluate the response
            bleu_score = sentence_bleu([expected_followup.split()], response.split())
            rouge_scores = rouge.get_scores(response, expected_followup, avg=True)
            rouge_l_f1 = rouge_scores['rouge-l']['f']
            cosine_similarity = compute_cosine_similarity(response, expected_followup)

            # Save the results
            results.append({
                'prompt_name': prompt_name,
                'question': question,
                'response': response,
                'bleu_score': bleu_score,
                'rouge_l_f1': rouge_l_f1,
                'cosine_similarity': cosine_similarity
            })

            print(f"\n{prompt_name} Evaluation:")
            print(f"BLEU Score: {bleu_score:.4f}")
            print(f"ROUGE-L F1 Score: {rouge_l_f1:.4f}")
            print(f"Cosine Similarity: {cosine_similarity:.4f}")
            print(f"Response:\n{response}\n")

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
    # Example question for follow-up experimentation
    # question = "Does this product come in different colors?"
    # state = {"question": question}

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
        "There is no mention of this product being available in different colors in the reviews or metadata.",
        "Does the product have any color customization options?\nWould you like to know about its design details?\nAre there similar products in different colors?"
    ),
    (
        "Is this product durable?",
        "Many reviews describe the product as durable and well-made, though a few users expected higher quality in details.",
        "Do you want more information about the product's build quality?\nWould you like to know how customers describe its longevity?\nShould I look for reviews mentioning its material strength?"
    ),
    (
        "Is this product lightweight?",
        "The product is lightweight and compact, as highlighted by several reviewers.",
        "Would you like to know its exact weight?\nShould I check if customers found it portable?\nDo you need details about its size and dimensions?"
    ),
    (
        "Can kids use it?",
        "The reviews and metadata do not specify whether the product is suitable for children, but it is generally described as a collectible rather than a toy.",
        "Do you want to know about its safety for kids?\nShould I check if customers bought it for children?\nWould you like to see similar items suitable for kids?"
    ),
    (
        "Does it come in different materials?",
        "The metadata and reviews do not mention if this product is available in different materials. It is made of vinyl as typical for FunKo POP! figures.",
        "Do you want details about the material used?\nShould I look for reviews discussing its material quality?\nWould you like to see other products with different materials?"
    ),
    (
        "What is the average rating of this product?",
        "The product has an average rating of 4.8 out of 5 based on 1,234 customer reviews.",
        "Do you want to see specific reviews from customers?\nWould you like details on the number of positive and negative ratings?\nShould I provide a summary of customer feedback?"
    ),
    (
        "Is it worth the price?",
        "Most reviewers consider the product worth the price due to its quality and collectible value, though a few felt it was slightly overpriced.",
        "Would you like to know more about customer opinions on pricing?\nShould I provide examples of value-added features?\nDo you want to compare it with similar products?"
    ),
    (
        "Does it come with a display box?",
        "Yes, the product comes in a window display box, as mentioned in the metadata and reviews.",
        "Do you want to know more about the box's quality?\nShould I find reviews discussing its packaging?\nWould you like to know about how it is shipped?"
    ),
    (
        "What do people like most about this product?",
        "Customers love the detailed craftsmanship, compact size, and nostalgic appeal of the product.",
        "Would you like more details about the product's design?\nShould I find reviews highlighting its best features?\nDo you want comparisons with similar collectibles?"
    ),
    (
        "Are there any issues with this product?",
        "Some reviewers noted that the product could have better detailing, and a few mentioned it being slightly overpriced.",
        "Would you like to see specific reviews mentioning issues?\nShould I summarize the negative feedback from customers?\nDo you want to know how common these issues are?"
    )
]


    # Prepare state
    state = MultiAgentState(
        question="",
        question_type="",
        answer="",
        meta_data=meta_df,
        retriever=retriever,
        )


    # Expected follow-up question
    expected_followup = "Would you like to know about its design details?\nDo you want details about similar products with color options?\nShould I find user reviews discussing color variants?"

    # Run the follow-up prompt experimentation
    best_prompt = followup_node_prompt_experimentation(state, questions_with_answers)
