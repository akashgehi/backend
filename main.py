from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pdf_processor import process_pdfs
from vector_store import VectorStore
from rag_chain import get_rag_response
import os
from dotenv import load_dotenv
import logging
from typing import List, Dict
import uuid

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

# In-memory chat history storage
chat_histories: Dict[str, List[Dict[str, str]]] = {}

@app.post("/upload-pdfs")
async def upload_pdfs(files: List[UploadFile] = File(...)):
    try:
        if not files:
            raise HTTPException(400, "No files uploaded")
        saved_paths = []
        os.makedirs("temp_uploads", exist_ok=True)
        for file in files:
            if not file.filename.lower().endswith(".pdf"):
                continue
            file_path = f"temp_uploads/{file.filename}"
            with open(file_path, "wb") as f:
                f.write(await file.read())
            saved_paths.append(file_path)
        documents = process_pdfs(saved_paths)
        vector_store.update_store(documents)
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
async def ask_question(data: Dict):
    try:
        if not data.get("question"):
            raise HTTPException(400, "Question field is required")
        if not data.get("session_id"):
            raise HTTPException(400, "Session ID is required")
        if not vector_store.db:
            raise HTTPException(400, "Please upload documents first")
        
        session_id = data["session_id"]
        question = data["question"]
        
        if session_id not in chat_histories:
            chat_histories[session_id] = []
        
        chat_histories[session_id].append({"role": "user", "content": question})
        
        response = get_rag_response(question, vector_store, chat_histories[session_id])
        
        chat_histories[session_id].append({"role": "assistant", "content": response["answer"]})
        
        if len(chat_histories[session_id]) > 20:
            chat_histories[session_id] = chat_histories[session_id][-20:]
        
        return {
            "answer": response["answer"],
            "sources": response["sources"],
            "usage": response["usage"],
            "session_id": session_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Question failed: {str(e)}")
        raise HTTPException(500, "Analysis service unavailable")

@app.post("/new-session")
async def new_session():
    session_id = str(uuid.uuid4())
    chat_histories[session_id] = []
    return {"session_id": session_id}

@app.get("/health")
async def health_check():
    doc_count = vector_store.db.index.ntotal if vector_store.db else 0
    return {
        "status": "online",
        "documents_loaded": vector_store.db is not None,
        "document_count": doc_count,
        "embedding_model": "all-mpnet-base-v2",
        "llm_service": "Google Gemini"
    }