from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import os
import logging

logger = logging.getLogger(__name__)

class VectorStore:
    def __init__(self, index_path: str = "faiss_index"):
        """Initialize VectorStore with persistent storage."""
        # Enhanced embeddings with GPU support
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2",
            model_kwargs={"device": "cuda" if os.getenv("USE_GPU") else "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
        self.db = None
        self.index_path = index_path
        
        # Load existing index if available
        self._load_index()

    def _load_index(self):
        """Load FAISS index from disk if it exists."""
        try:
            if os.path.exists(self.index_path):
                logger.info(f"Loading existing vector store from {self.index_path}")
                self.db = FAISS.load_local(
                    self.index_path,
                    self.embeddings,
                    allow_dangerous_deserialization=True  # Required for FAISS deserialization
                )
                logger.info(f"Loaded vector store with {self.db.index.ntotal} embeddings")
            else:
                logger.info("No existing index found, starting with empty vector store")
        except Exception as e:
            logger.error(f"Failed to load vector store from {self.index_path}: {str(e)}")
            self.db = None

    def update_store(self, documents):
        """Update vector store and save to disk."""
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
                
            # Save the updated index to disk
            self._save_index()
                
            logger.info(f"Vector store now contains {self.db.index.ntotal} embeddings")
            
        except Exception as e:
            logger.error(f"Vector store update failed: {str(e)}")
            raise

    def _save_index(self):
        """Save FAISS index to disk."""
        try:
            os.makedirs(self.index_path, exist_ok=True)
            self.db.save_local(self.index_path)
            logger.info(f"Vector store saved to {self.index_path}")
        except Exception as e:
            logger.error(f"Failed to save vector store to {self.index_path}: {str(e)}")
            raise