# from langchain_community.embeddings.ollama import OllamaEmbeddings
# from langchain_community.embeddings.bedrock import BedrockEmbeddings


# def get_embedding_function():
#     # Comment out BedrockEmbeddings if it's causing issues
#     # embeddings = BedrockEmbeddings(
#     #     credentials_profile_name="default", region_name="us-east-1"
#     # )
    
#     # Use OllamaEmbeddings instead
#     embeddings = OllamaEmbeddings(model="nomic-embed-text")
#     return embeddings







# from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

class ChromaEmbeddingFunction:
    def __init__(self, embeddings):
        self.embeddings = embeddings

    def __call__(self, input):
        return self.embed_query(input)

    def embed_query(self, query):
        """Convert the query to an embedding vector."""
        return self.embeddings.embed_query(query)

    def embed_documents(self, texts):
        """Convert documents to embedding vectors."""
        return self.embeddings.embed_documents(texts)

def get_embedding_function():
    # Pass the model name as a string
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    return ChromaEmbeddingFunction(embeddings)

