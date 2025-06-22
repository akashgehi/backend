import os

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
VECTOR_DB_PATH = "vector_store"
EMBED_MODEL = "text-embedding-ada-002"
