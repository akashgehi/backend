from langchain_community.document_loaders import PyPDFLoader
import pdfplumber
from typing import List
from langchain.schema import Document
import logging
import re

logger = logging.getLogger(__name__)

def clean_text(text: str) -> str:
    """Enhanced text cleaning for PDF content"""
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    # Fix hyphenated words
    text = re.sub(r'(\w+)-\s+(\w+)', r'\1\2', text)
    return text.strip()

def process_pdfs(pdf_paths: List[str]) -> List[Document]:
    documents = []
    
    for path in pdf_paths:
        try:
            with pdfplumber.open(path) as pdf:
                for i, page in enumerate(pdf.pages):
                    text = page.extract_text()
                    if text:
                        # Enhanced metadata per page
                        metadata = {
                            "source": path.split("/")[-1],
                            "page": i + 1,
                            "total_pages": len(pdf.pages),
                            "author": pdf.metadata.get("Author", ""),
                            "title": pdf.metadata.get("Title", path.split("/")[-1])
                        }
                        
                        # Clean and chunk each page individually
                        cleaned_text = clean_text(text)
                        from rag_chain import text_splitter
                        page_chunks = text_splitter.create_documents(
                            [cleaned_text],
                            [metadata]
                        )
                        documents.extend(page_chunks)
                        
        except Exception as e:
            logger.error(f"Failed to process {path}: {str(e)}")
            continue
    
    if not documents:
        raise ValueError("No valid documents were processed from the provided PDFs")
    
    return documents