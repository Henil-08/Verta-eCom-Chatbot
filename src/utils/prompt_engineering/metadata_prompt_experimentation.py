# metadata_prompt_experimentation.py
import os
from dotenv import load_dotenv
import pandas as pd
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from state import MultiAgentState
from langchain.schema import Document
from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge
from helpers import compute_cosine_similarity

os.environ['GROQ_API_KEY']=os.getenv("GROQ_API_KEY")

def metadata_prompt_experimentation(state, expected_summary):
    """
    Performs prompt experimentation for the Metadata Agent, evaluates each response,
    and selects the best prompt based on evaluation metrics.

    Args:
        state (MultiAgentState): The current state containing metadata.
        expected_summary (str): The expected summary for evaluation.

    Returns:
        best_prompt (dict): The best prompt and its evaluation metrics.
    """
    meta_llm = ChatGroq(model_name="llama-3.1-8b-instant")
    meta_df = state['meta_data']

    # Prepare the metadata
    meta_data = {
        'main_category': meta_df.at[0, 'main_category'],
        'title': meta_df.at[0, 'title'],
        'average_rating': meta_df.at[0, 'average_rating'],
        'rating_number': meta_df.at[0, 'rating_number'],
        'features': meta_df.at[0, 'features'],
        'description': meta_df.at[0, 'description'],
        'price': meta_df.at[0, 'price'],
        'store': meta_df.at[0, 'store'],
        'categories': meta_df.at[0, 'categories'],
        'details': meta_df.at[0, 'details'],
        'full_details': meta_df.iloc[0].to_string()
    }

    # Improved prompts
    prompts = [
        ('Prompt 1', '''
            You are a great Data Interpreter and Summarizer. Read the Product Meta Data sent to you and produce it in 500 words.

            Meta Data:
            main_category: {main_category}
            title: {title}
            average_rating: {average_rating}
            rating_number: {rating_number}
            features: {features}
            description: {description}
            price: {price}
            store: {store}
            categories: {categories}
            details: {details}

            Return in a proper format:
            main_category: Same 
            title: Same
            average_rating: Same
            rating_number: Same
            features: Summarize 
            description: Summarize
            price: Same
            store: Same 
            categories: Same 
            details: Same/Summarize where necessary 

            Do not answer any user question, just provide the meta data
        '''),
        ('Prompt 2', '''
            As a product expert, provide a concise and informative summary of the product described below. Focus on highlighting the main features, specifications, and any unique selling points. Your summary should be less than 500 words.

            Product Information:
            main_category: {main_category}
            title: {title}
            average_rating: {average_rating}
            rating_number: {rating_number}
            features: {features}
            description: {description}
            price: {price}
            store: {store}
            categories: {categories}
            details: {details}

            Return in a proper format:
            main_category: Same 
            title: Same
            average_rating: Same
            rating_number: Same
            features: Summarize 
            description: Summarize
            price: Same
            store: Same 
            categories: Same 
            details: Same/Summarize where necessary 

            Please present the information in clear, professional language suitable for potential customers.
        '''),
        ('Prompt 3', '''
            You are a professional content writer tasked with creating an engaging and informative product overview based on the data provided below. Your summary should highlight the key features, benefits, and any standout details that would appeal to potential buyers. Limit your summary to 500 words.

            Product Data:
            main_category: {main_category}
            title: {title}
            average_rating: {average_rating}
            rating_number: {rating_number}
            features: {features}
            description: {description}
            price: {price}
            store: {store}
            categories: {categories}
            details: {details}

            Return in a proper format:
            main_category: Same 
            title: Same
            average_rating: Same
            rating_number: Same
            features: Summarize 
            description: Summarize
            price: Same
            store: Same 
            categories: Same 
            details: Same/Summarize where necessary 
            Ensure the summary is well-structured and easy to read.
        '''),
        ('Prompt 4', '''
            As an AI assistant, generate a detailed yet concise summary of the following product. Focus on the main features, specifications, and any notable aspects that make the product unique. Keep the summary under 500 words.

            Product Details:
            main_category: {main_category}
            title: {title}
            average_rating: {average_rating}
            rating_number: {rating_number}
            features: {features}
            description: {description}
            price: {price}
            store: {store}
            categories: {categories}
            details: {details}

            Return in a proper format:
            main_category: Same 
            title: Same
            average_rating: Same
            rating_number: Same
            features: Summarize 
            description: Summarize
            price: Same
            store: Same 
            categories: Same 
            details: Same/Summarize where necessary 

            Your summary should be written in third person and be helpful to someone considering purchasing the product.
        '''),
        ('Prompt 5', '''
            Create a customer-friendly product summary using the information below. Emphasize the key features, benefits, and why this product stands out in its category. The summary should be less than 500 words.

            Product Information:
            main_category: {main_category}
            title: {title}
            average_rating: {average_rating}
            rating_number: {rating_number}
            features: {features}
            description: {description}
            price: {price}
            store: {store}
            categories: {categories}
            details: {details}

            Return in a proper format:
            main_category: Same 
            title: Same
            average_rating: Same
            rating_number: Same
            features: Summarize 
            description: Summarize
            price: Same
            store: Same 
            categories: Same 
            details: Same/Summarize where necessary 

            Write in a clear and engaging style suitable for a product description on an e-commerce website.
        '''),
        ('Prompt 6', '''
            You are an expert in product descriptions. Using the data provided, write a compelling summary that highlights the product's main features, benefits, and any unique selling points. Your summary should be less than 500 words and aimed at helping potential customers make an informed decision.

            Product Data:
            main_category: {main_category}
            title: {title}
            average_rating: {average_rating}
            rating_number: {rating_number}
            features: {features}
            description: {description}
            price: {price}
            store: {store}
            categories: {categories}
            details: {details}

            Return in a proper format:
            main_category: Same 
            title: Same
            average_rating: Same
            rating_number: Same
            features: Summarize 
            description: Summarize
            price: Same
            store: Same 
            categories: Same 
            details: Same/Summarize where necessary 

        Ensure the summary is professional, informative, and persuasive.
        '''),
        (
            'Prompt 7', '''
            You are an expert in creating structured, concise, and informative product summaries for e-commerce listings. Review the Product Meta Data provided and generate a high-quality summary that highlights the product's main features, unique selling points, and overall value to potential buyers. Limit the response to 500 words or fewer.

            Meta Data:
            main_category: {main_category}
            title: {title}
            average_rating: {average_rating}
            rating_number: {rating_number}
            features: {features}
            description: {description}
            price: {price}
            store: {store}
            categories: {categories}
            details: {details}

            Instructions:
            1. Focus on conveying the product's main features and unique qualities, such as materials, dimensions, or special functions.
            2. Emphasize any specific attributes that would make this product appealing to collectors, enthusiasts, or fans.
            3. Ensure each section is summarized concisely:
                - main_category: As provided
                - title: As provided
                - average_rating: As provided
                - rating_number: As provided
                - features: Summarize to highlight the key qualities and distinct attributes.
                - description: Summarize to provide an engaging overview without redundant details.
                - price: As provided
                - store: As provided
                - categories: As provided
                - details: Summarize where necessary, focusing on the most relevant specs (e.g., dimensions, weight, materials).
            4. Do not answer any user questions, provide opinions, or add extraneous information. Only include the metadata.

            Return the result in a clear, structured format, strictly adhering to the fields and order provided.
        '''
        )
    ]

    results = []
    rouge = Rouge()

    for prompt_name, prompt_template in prompts:
        # Format the prompt with metadata
        prompt = prompt_template.format(**meta_data)

        # Build the prompt
        meta_qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", prompt),
            ]
        )
        parser = StrOutputParser()
        meta_chain = meta_qa_prompt | meta_llm | parser

        try:
            # Generate summary
            meta_results = meta_chain.invoke({'input': ''})
            response = meta_results
        except Exception as error:
            print(f"Error with {prompt_name}: {error}")
            response = "Error generating response."

        # Evaluate the response
        bleu_score = sentence_bleu([expected_summary.split()], response.split())
        rouge_scores = rouge.get_scores(response, expected_summary, avg=True)
        rouge_l_f1 = rouge_scores['rouge-l']['f']
        cosine_similarity = compute_cosine_similarity(response, expected_summary)

        # Save the results
        results.append({
            'prompt_name': prompt_name,
            'prompt': prompt,
            'response': response,
            'bleu_score': bleu_score,
            'rouge_l_f1': rouge_l_f1,
            'cosine_similarity': cosine_similarity
        })

        print(f"\n{prompt_name} Evaluation:")
        print(f"BLEU Score: {bleu_score:.4f}")
        print(f"ROUGE-L F1 Score: {rouge_l_f1:.4f}")
        print(f"Cosine Similarity: {cosine_similarity:.4f}")
        # print(f"Response:\n{response}\n")

    # Determine the best prompt based on ROUGE-L F1 Score
    best_prompt = max(results, key=lambda x: x['cosine_similarity'])

    print(f"\nBest Prompt: {best_prompt['prompt_name']} with ROUGE-L F1 Score: {best_prompt['rouge_l_f1']:.4f}")
    # print(f"Prompt:\n{best_prompt['prompt']}")
    # print(f"Response:\n{best_prompt['response']}")
    print(f"Cosine Similarity: {best_prompt['cosine_similarity']:.4f}")

    return best_prompt

if __name__ == "__main__":
    # Load meta_df and prepare state
    # Replace with actual metadata loading
    # For example, let's assume we have a meta_df loaded from a CSV or database
    # meta_df = pd.read_csv('meta_df.csv')
    # For demonstration, let's create a sample meta_df
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

    # Prepare state
    state = MultiAgentState(
        question="What are the main features of this product?",
        question_type="",
        answer="",
        documents=[],
        meta_data=meta_df,
        retriever=None,
        followup_questions=None
    )

    # After importing MultiAgentState
    print(f"MultiAgentState imported from: {MultiAgentState.__module__}")

    # After creating 'state'
    print(f"Type of 'state': {type(state)}")

    expected_summary = '''
        Main Category: Toys & Games
        Title: FunKo POP! Movies Lord of the Rings Frodo Baggins Vinyl Figure
        Average Rating: 4.8 out of 5
        Rating Number: 1,234 reviews

        Features:
        This FunKo POP! figure is a stylized collectible that stands 3.75 inches tall, depicting the beloved character Frodo Baggins from the iconic Lord of the Rings series. With FunKo’s classic oversized head and large round eyes, the figure has a whimsical and endearing look that resonates with fans of both the series and FunKo's collectible style. It is an ideal item for Lord of the Rings enthusiasts or FunKo POP! collectors, featuring detailed craftsmanship that captures Frodo’s distinct character traits.

        Description:
        From the popular Lord of the Rings movie series, this FunKo POP! Vinyl Figure presents Frodo Baggins, the hero of Middle-earth, in FunKo's signature style. The figure arrives in a protective window display box, making it suitable for showcasing in or out of the box, depending on collector preferences. The display box also includes Lord of the Rings and FunKo POP! branding, enhancing its value for collectors who prioritize authenticity. The lightweight design makes it easy to handle or reposition on shelves or in display cases.

        Price: $10.99
        Store: FunKo Store
        Categories: Collectibles, Vinyl Figures

        Details:

        Product Dimensions: 2.5 x 2.5 x 3.75 inches
        Item Weight: 4 ounces
        Manufacturer: FunKo
        Frodo Baggins, as a character, is known for his bravery, resilience, and compassion, all of which are symbolized in this collectible item. The figurine's compact size (3.75 inches tall) makes it an excellent display piece that can fit easily on bookshelves, desks, or within glass display cabinets. Its weight of only 4 ounces ensures it is sturdy yet lightweight, ideal for collectors who may want to rearrange their items frequently.

        This FunKo POP! figure has earned an impressive 4.8-star rating out of 5 on popular review platforms, based on 1,234 customer reviews. The positive feedback underscores its popularity and high quality, with reviewers frequently mentioning its durability, aesthetic appeal, and the nostalgia it evokes for fans of the Lord of the Rings trilogy. Customers particularly appreciate the attention to detail in Frodo’s attire and expression, as well as the collectible value of having this character immortalized in vinyl form. FunKo’s POP! Movies series has become a cherished part of many collections, celebrated for capturing cultural icons in an approachable, fun format.

        Priced at $10.99, this figure offers excellent value for its quality, design, and collectible nature. It is available from the FunKo Store, a trusted source for authentic FunKo merchandise. Collectors often seek figures in FunKo's POP! Movies line due to its emphasis on beloved characters from movies, and Frodo Baggins is a fitting addition for any fan of Middle-earth.

        The FunKo POP! Movies Lord of the Rings Frodo Baggins Vinyl Figure is not just a toy; it’s a tribute to one of literature and film's most cherished characters. Whether as a standalone collectible or as part of a broader Lord of the Rings or FunKo collection, this Frodo figure is bound to bring joy to its owner. Its accessible price, high quality, and unique design make it a must-have for collectors and a delightful gift for fans of the Lord of the Rings series.

        '''
        
    best_prompt = metadata_prompt_experimentation(state, expected_summary)
