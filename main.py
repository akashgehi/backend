from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pdf_processor import process_pdfs
from vector_store import VectorStore
from rag_chain import get_rag_response
import os
from dotenv import load_dotenv
import logging
from typing import List

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Enhanced RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize vector store
vector_store = VectorStore()

@app.post("/upload-pdfs")
async def upload_pdfs(files: List[UploadFile] = File(...)):
    """Enhanced PDF upload endpoint with validation"""
    try:
        if not files:
            raise HTTPException(400, "No files uploaded")
            
        # Save files temporarily
        saved_paths = []
        os.makedirs("temp_uploads", exist_ok=True)
        
        for file in files:
            if not file.filename.lower().endswith(".pdf"):
                continue
                
            file_path = f"temp_uploads/{file.filename}"
            with open(file_path, "wb") as f:
                f.write(await file.read())
            saved_paths.append(file_path)
        
        # Process and store
        documents = process_pdfs(saved_paths)
        vector_store.update_store(documents)
        
        # Cleanup
        for path in saved_paths:
            os.remove(path)
            
        return {
            "status": "success",
            "documents_processed": len(documents),
            "average_chunk_size": sum(len(d.page_content) for d in documents)/len(documents)
        }
        
    except Exception as e:
        logger.error(f"Upload failed: {str(e)}")
        raise HTTPException(500, f"Processing error: {str(e)}")

@app.post("/ask")
async def ask_question(question: dict):
    """Enhanced question endpoint with proper validation"""
    try:
        if not question.get("question"):
            raise HTTPException(400, "Question field is required")
            
        if not vector_store.db:
            raise HTTPException(400, "Please upload documents first")
            
        return get_rag_response(question["question"], vector_store)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Question failed: {str(e)}")
        raise HTTPException(500, "Analysis service unavailable")

@app.get("/health")
async def health_check():
    """Enhanced health check with system status"""
    return {
        "status": "online",
        "documents_loaded": vector_store.db is not None,
        "embedding_model": "all-mpnet-base-v2",
        "llm_service": "Google Gemini"
    }