from openai import OpenAI
from rag.vector_store import create_faiss_index, query_index
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def embed_chunks(text, chunk_size=500):
    chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
    response = client.embeddings.create(
        input=chunks,
        model="text-embedding-3-small"
    )
    vectors = [r.embedding for r in response.data]
    index = create_faiss_index(vectors)
    return chunks, index

def get_relevant_answer(question, chunks, index):
    response = client.embeddings.create(
        input=[question],
        model="text-embedding-3-small"
    )
    query_vector = response.data[0].embedding
    best_idx = query_index(index, query_vector)
    context = chunks[best_idx]
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": context}, {"role": "user", "content": question}]
    )
    return completion.choices[0].message.content
