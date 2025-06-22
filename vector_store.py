from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import os
import logging

logger = logging.getLogger(__name__)

class VectorStore:
    def __init__(self):
        # Enhanced embeddings with GPU support
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2",
            model_kwargs={"device": "cuda" if os.getenv("USE_GPU") else "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
        self.db = None
        
    def update_store(self, documents):
        try:
            if not documents:
                raise ValueError("Empty document list provided")
                
            if self.db is None:
                logger.info("Creating new vector store with enhanced configuration")
                self.db = FAISS.from_documents(
                    documents,
                    self.embeddings,
                    normalize_L2=True  # Better similarity search
                )
            else:
                logger.info("Adding documents to existing vector store")
                self.db.add_documents(documents)
                
            logger.info(f"Vector store now contains {self.db.index.ntotal} embeddings")
            
        except Exception as e:
            logger.error(f"Vector store update failed: {str(e)}")
            raise