"""
rag_embedding.py
Embedding file for RAG memory access
"""
import csv
import ollama
import chromadb

from conf_module import load_conf

MODEL=load_conf("EMBED_MODEL")

client = chromadb.Client()
collection = client.create_collection(name="docs")

def read_memory(n, user, query) -> list:
    """
    Read memory from CSV using RAG
    
    Args:
        n (int): Number of relevant documents to retrieve
        user (str): The user to whom the memory relates
        query (str): The query to search relevant documents
    
    Returns:
        list: List of relevant documents
    """

    input = f"{user} {query}"
    # Load documents from CSV
    documents = []
    with open("data.csv", newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            documents.append(f"{row['user']} {row['content']}")

    # Embed and store each document
    for i, d in enumerate(documents):
        response = ollama.embed(model=MODEL, input=d)
        embeddings = response["embeddings"]
        collection.add(
            ids=[str(i)],
            embeddings=embeddings,
            documents=[d]
        )

    # Generate embedding for input and search relevant document
    response = ollama.embed(model=MODEL, input=input)
    results = collection.query(
        query_embeddings=[response["embeddings"][0]],
        n_results=n
    )
    data = results['documents'][0]
    return(data)

def write_memory(user: str, content: str) -> None:
    """
    Write memory to CSV

    Args:
        user (str): The user to whom the memory relates
        content (str): The content to store
    
    Returns: None
    """
    fieldnames = ["user", "content"]
    try:
        # Open CSV in append mode
        with open("data.csv", "a", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            # If file is empty, write header
            if csvfile.tell() == 0:
                writer.writeheader()
            writer.writerow({"user": user, "content": content})
    except Exception as e:
        print(f"[write_memory] Error writing memory: {e}")