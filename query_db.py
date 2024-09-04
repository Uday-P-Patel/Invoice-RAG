import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from sentence_transformers import CrossEncoder
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from get_embedding_function import get_embedding_function
from huggingface_hub import login
import yaml
from tqdm import tqdm  # For progress tracking

CHROMA_PATH = "chroma"

# PROMPT_TEMPLATE = """
# Answer the question based only on the following context:
# {context}
# ---
# Question: {question}
# Given the following context, please answer the question . If the answer is not in the context, say "I don't have enough information to answer that question."

# Answer:
# """

PROMPT_TEMPLATE = """
You are provided with data from invoices. Answer the following question based solely on this invoice data:

Context:
{context}
---
Question: {question}
Given the provided invoice data, answer the question as accurately and concisely as possible. If the information needed to answer the question is not available in the context, respond with "The invoice data does not provide enough information to answer this question."

Answer:
"""

def load_config():
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)

config = load_config()

# Hugging Face login
login(token=config['hf_token'])

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model and tokenizer initialization
quantization_config = BitsAndBytesConfig(load_in_4bit=True)
tokenizer = AutoTokenizer.from_pretrained(config['model_path'])
model = AutoModelForCausalLM.from_pretrained(
    config['model_path'],
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)

# Reranker initialization
# reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-12-v2')

def rerank_documents(question, docs, k=5):
    pairs = [[question, doc.page_content] for doc in docs]
    scores = reranker.predict(pairs)
    reranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in reranked[:k]]

def generate_answer(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=config['max_new_tokens'],
            num_return_sequences=1,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )
    # Decoding and extracting only the answer part
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Remove anything before the actual answer starts, if the model includes the prompt
    answer_start = answer.find("Answer:")
    if answer_start != -1:
        answer = answer[answer_start + len("Answer:"):].strip()

    return answer

def post_process_answer(answer):
    # Implement post-processing logic here if needed
    return answer.strip()

def query_rag(query_text: str, db):
    # Batch size for processing
    batch_size = 1000

    # Get all the vectors from the database
    all_results = []
    for i in tqdm(range(0, db._chroma_collection.count(), batch_size), desc="Processing batches"):
        # Fetch a batch of results
        batch_results = db.similarity_search_with_score(query_text, k=10)
        all_results.extend(batch_results)

    # Filter top k results based on the score
    # reranked_docs = rerank_documents(query_text, [doc for doc, _ in all_results], k=config['k_relevant_docs'])
    reranked_docs = rerank_documents(query_text, [doc for doc, _ in all_results], k=db._chroma_collection.count())
    
    context_text = "\n\n---\n\n".join([doc.page_content for doc in reranked_docs])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    
    # Generate the answer using the model
    response_text = generate_answer(prompt)
    refined_response = post_process_answer(response_text)
    
    
    # Print only the answer and the primary source file
    print(f"Answer: {refined_response}")
    
    # Extract sources from the reranked documents
    sources = [doc.metadata.get("source", "Unknown Source") for doc in reranked_docs]
    print(f"Source Files: {', '.join(sources)}")
    
    # # Extract the most relevant source from the first document in the reranked list
    # primary_source = reranked_docs[0].metadata.get("source", "Unknown Source")
    # print(f"Primary Source File: {primary_source}")

    return refined_response

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text
    
    # Prepare the DB (Chroma with efficient search setup)
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    
    query_rag(query_text, db)

if __name__ == "__main__":
    main()


















# import argparse
# import torch
# from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
# from sentence_transformers import CrossEncoder
# from langchain_chroma import Chroma
# from langchain.prompts import ChatPromptTemplate
# from get_embedding_function import get_embedding_function
# from huggingface_hub import login
# import yaml

# CHROMA_PATH = "chroma"

# PROMPT_TEMPLATE = """
# Answer the question based only on the following context:
# {context}
# ---
# Question: {question}
# Given the following context, please answer the question. If the answer is not in the context, say "I don't have enough information to answer that question."

# Answer:
# """

# def load_config():
#     with open('config.yaml', 'r') as f:
#         return yaml.safe_load(f)

# config = load_config()

# # Hugging Face login
# login(token=config['hf_token'])

# # Device setup
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Model and tokenizer initialization
# quantization_config = BitsAndBytesConfig(load_in_4bit=True)
# tokenizer = AutoTokenizer.from_pretrained(config['model_path'])
# model = AutoModelForCausalLM.from_pretrained(
#     config['model_path'],
#     # quantization_config=quantization_config,
#     torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
#     device_map="auto"
# )

# # Reranker initialization
# reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
# # reranker = CrossEncoder('cross-encoder/ms-marco-TinyBERT-L-2-v2')

# def rerank_documents(question, docs, k=5):
#     pairs = [[question, doc.page_content] for doc in docs]
#     scores = reranker.predict(pairs)
#     reranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
#     return [doc for doc, _ in reranked[:k]]

# # def generate_answer(prompt):
# #     inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
# #     with torch.no_grad():
# #         outputs = model.generate(
# #             **inputs,
# #             max_new_tokens=config['max_new_tokens'],
# #             num_return_sequences=1,
# #             do_sample = True,
# #             temperature=0.7,
# #             top_p=0.9,
# #         )
# #     return tokenizer.decode(outputs[0], skip_special_tokens=True)

# def generate_answer(prompt):
#     inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
#     with torch.no_grad():
#         outputs = model.generate(
#             **inputs,
#             max_new_tokens=config['max_new_tokens'],
#             num_return_sequences=1,
#             do_sample=True,
#             temperature=0.7,
#             top_p=0.9,
#         )
#     # Decoding and extracting only the answer part
#     answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

#     # Remove anything before the actual answer starts, if the model includes the prompt
#     answer_start = answer.find("Answer:")
#     if answer_start != -1:
#         answer = answer[answer_start + len("Answer:"):].strip()

#     return answer


# def post_process_answer(answer):
#     # Implement post-processing logic here if needed
#     return answer.strip()

# def query_rag(query_text: str):
#     # Prepare the DB
#     embedding_function = get_embedding_function()
#     db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    
#     # Search the DB
#     results = db.similarity_search_with_score(query_text, k=10)
#     reranked_docs = rerank_documents(query_text, [doc for doc, _ in results], k=config['k_relevant_docs'])
    
#     context_text = "\n\n---\n\n".join([doc.page_content for doc in reranked_docs])
#     prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
#     prompt = prompt_template.format(context=context_text, question=query_text)
    
#     # Generate the answer using the model
#     response_text = generate_answer(prompt)
#     refined_response = post_process_answer(response_text)
    
    # # Extract sources from the reranked documents
    # sources = [doc.metadata.get("source", "Unknown Source") for doc in reranked_docs]
    
    # # Print only the answer and the source files
    # print(f"Answer: {refined_response}")
    # print(f"Source Files: {', '.join(sources)}")

#     return refined_response

# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("query_text", type=str, help="The query text.")
#     args = parser.parse_args()
#     query_text = args.query_text
#     query_rag(query_text)

# if __name__ == "__main__":
#     main()