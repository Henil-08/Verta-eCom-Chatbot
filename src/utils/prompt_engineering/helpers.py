from sentence_transformers import SentenceTransformer, util

def compute_cosine_similarity(expected, generated):
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    expected_embedding = model.encode(expected, convert_to_tensor=True)
    generated_embedding = model.encode(generated, convert_to_tensor=True)
    similarity_score = util.pytorch_cos_sim(expected_embedding, generated_embedding)
    return similarity_score.item()

# import openai
# import time

# def call_openai_with_retries(model, messages, max_retries=3):
#     retries = 0
#     while retries < max_retries:
#         try:
#             response = openai.chat.completions.create(
#                 model=model,
#                 messages=messages
#             )
#             return response
#         except Exception as e:
#             print(f"Error: {e}. Retrying {retries + 1}/{max_retries}...")
#             retries += 1
#             time.sleep(2)  # Wait briefly before retrying
#     print("Failed after retries.")
#     return None

# # Usage example
# response = call_openai_with_retries(
#     model="gpt-4o-mini",
#     messages=[{"role": "user", "content": "Hello, can you confirm model access?"}]
# )
# for chunk in response:
#     print(chunk)
