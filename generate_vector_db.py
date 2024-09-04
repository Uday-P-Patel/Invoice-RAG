# import argparse
# import os
# import shutil
# from langchain.document_loaders import PyPDFDirectoryLoader, TextLoader, UnstructuredWordDocumentLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.schema.document import Document
# from langchain.vectorstores.chroma import Chroma
# from get_embedding_function import get_embedding_function

# CHROMA_PATH = "chroma"
# DATA_PATH = "data"

# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--reset", action="store_true", help="Reset the database.")
#     args = parser.parse_args()
    
#     if args.reset:
#         print("✨ Clearing Database")
#         clear_database()

#     print("📄 Loading documents...")
#     documents = load_documents()
#     print("✂️ Splitting documents into chunks...")
#     chunks = split_documents(documents)
#     print("📦 Adding chunks to vector store...")
#     add_to_chroma(chunks)

# def load_documents():
#     documents = []
#     for root, _, files in os.walk(DATA_PATH):
#         for file in files:
#             file_path = os.path.join(root, file)
#             _, file_extension = os.path.splitext(file)
            
#             if file_extension.lower() == '.pdf':
#                 loader = PyPDFDirectoryLoader(file_path)
#             elif file_extension.lower() in ['.txt']:
#                 loader = TextLoader(file_path)
#             elif file_extension.lower() in ['.doc', '.docx']:
#                 loader = UnstructuredWordDocumentLoader(file_path)
#             else:
#                 print(f"Unsupported file type: {file_extension}")
#                 continue
            
#             documents.extend(loader.load())
    
#     return documents

# def split_documents(documents: list[Document]):
#     text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=800,
#         chunk_overlap=80
#     )
#     return text_splitter.split_documents(documents)

# def add_to_chroma(chunks: list[Document]):
#     db = Chroma(
#         persist_directory=CHROMA_PATH,
#         embedding_function=get_embedding_function()
#     )
    
#     chunks_with_ids = calculate_chunk_ids(chunks)
#     existing_items = db.get(include=[])
#     existing_ids = set(existing_items["ids"])
#     print(f"Number of existing documents in DB: {len(existing_ids)}")

#     new_chunks = [chunk for chunk in chunks_with_ids if chunk.metadata["id"] not in existing_ids]
    
#     if len(new_chunks):
#         print(f"👉 Adding new documents: {len(new_chunks)}")
#         new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
#         db.add_documents(new_chunks, ids=new_chunk_ids)
#         db.persist()
#     else:
#         print("✅ No new documents to add")

# def calculate_chunk_ids(chunks):
#     last_page_id = None
#     current_chunk_index = 0

#     for chunk in chunks:
#         source = chunk.metadata.get("source")
#         current_page_id = f"{source}"

#         if current_page_id == last_page_id:
#             current_chunk_index += 1
#         else:
#             current_chunk_index = 0

#         chunk_id = f"{current_page_id}:{current_chunk_index}"
#         last_page_id = current_page_id
#         chunk.metadata["id"] = chunk_id

#     return chunks

# def clear_database():
#     if os.path.exists(CHROMA_PATH):
#         shutil.rmtree(CHROMA_PATH)

# if __name__ == "__main__":
#     main()


















import argparse
import asyncio
import hashlib
import os
import shutil
from langchain_community.document_loaders import PyPDFDirectoryLoader, TextLoader, UnstructuredWordDocumentLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from langchain_chroma import Chroma
# from langchain_community.vectorstores.chroma import Chroma
from get_embedding_function import get_embedding_function
from PIL import Image
import easyocr
# import pytesseract

CHROMA_PATH = "chroma"

def process_image(file_path: str) -> str:
    """Process image files and return text content using OCR."""
    try:
        image = Image.open(file_path)
        reader = easyocr.Reader(['en'])
        text_list = reader.readtext(file_path, detail=0)
        return ' '.join(text_list)
    except Exception as e:
        print(f"Error processing image {file_path}: {str(e)}")
        return ""

def convert_tiff_to_png(tiff_path: str, output_path: str) -> str:
    """Convert TIFF to PNG format."""
    try:
        with Image.open(tiff_path) as img:
            img = img.convert("RGB")
            img.save(output_path, format="PNG")
        return output_path
    except Exception as e:
        print(f"Error converting TIFF to PNG: {str(e)}")
        return None

async def process_file(file_path: str) -> str:
    """Process different file types and return text content."""
    _, file_extension = os.path.splitext(file_path)
    
    if file_extension.lower() == '.pdf':
        loader = PyPDFDirectoryLoader(file_path)
        pages = loader.load()
        content = ' '.join([page.page_content for page in pages])
        return content
    
    elif file_extension.lower() in ['.png', '.jpg', '.jpeg']:
        return process_image(file_path)
    
    elif file_extension.lower() == '.tif':
        converted_file = convert_tiff_to_png(file_path, "output.png")
        if converted_file:
            return process_image(converted_file)
        else:
            return ""
    
    elif file_extension.lower() == '.txt':
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    
    elif file_extension.lower() in ['.doc', '.docx']:
        loader = UnstructuredWordDocumentLoader(file_path)
        doc = loader.load()
        return doc[0].page_content
    
    elif file_extension.lower() in ['.db']:
        return ''
    
    
    
    else:
        raise ValueError(f"Unsupported file type: {file_extension}")

async def load_documents_from_folder(folder_path: str) -> list:
    """Load and process all documents from the given folder path."""
    documents = []
    for root, _, files in os.walk(folder_path):
        print(files)
        for file in files:
            file_path = os.path.join(root, file)
            text = await process_file(file_path)
            if text:
                documents.append(Document(page_content=text, metadata={"source": file_path}))
    return documents

def split_documents(documents: list[Document]) -> list[Document]:
    """Split documents into chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)

def generate_unique_id(file_path: str, chunk_number: int) -> str:
    """Generate a unique ID for each document chunk."""
    return hashlib.sha256(f"{file_path}_{chunk_number}".encode()).hexdigest()

def add_to_chroma(chunks: list[Document]):
    """Add chunks to the Chroma vector store."""
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=get_embedding_function())
    existing_items = db.get(include=[])
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    new_chunks = []
    new_chunk_ids = []
    
    for i, chunk in enumerate(chunks):
        chunk_id = generate_unique_id(chunk.metadata["source"], i)
        if chunk_id not in existing_ids:
            new_chunks.append(chunk)
            new_chunk_ids.append(chunk_id)
    
    if new_chunks:
        print(f"👉 Adding new documents: {len(new_chunks)}")
        db.add_documents(new_chunks, ids=new_chunk_ids)
        db.persist()
    else:
        print("✅ No new documents to add")

def clear_database():
    """Clear the Chroma database."""
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

def main(folder_path: str, reset: bool):
    """Main function to create or update the vector database."""
    if reset:
        print("✨ Clearing Database")
        clear_database()

    documents = asyncio.run(load_documents_from_folder(folder_path))
    chunks = split_documents(documents)
    add_to_chroma(chunks)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a vector database from documents in a folder.")
    parser.add_argument("folder_path", help="Path to the folder containing files")
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    args = parser.parse_args()

    main(args.folder_path, args.reset)