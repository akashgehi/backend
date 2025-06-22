from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from typing import List, Dict, Any
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Text splitter configuration
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
    add_start_index=True
)

def format_docs(docs: List[Document]) -> str:
    return "\n\n".join(
        f"Document {i+1} (Page {doc.metadata.get('page', '?')}):\n{doc.page_content}"
        for i, doc in enumerate(docs)
    )

def get_rag_response(question: str, vector_store) -> Dict[str, Any]:
    try:
        # Verify API key is loaded
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")

        # Initialize LLM with minimal configuration
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=api_key,
            temperature=0.3
        )

        # Simple prompt template
        prompt = ChatPromptTemplate.from_template("""
        Answer the question based on this context:
        
        {context}
        
        Question: {question}
        
        Answer concisely in 2-3 sentences:""")

        # Basic retrieval
        retriever = vector_store.db.as_retriever(search_kwargs={"k": 3})

        chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        # Get answer and sources
        answer = chain.invoke(question)
        docs = vector_store.db.similarity_search(question, k=3)
        
        return {
            "answer": answer,
            "sources": [{
                "source": doc.metadata.get("source", "Unknown"),
                "page": doc.metadata.get("page", "N/A")
            } for doc in docs]
        }

    except Exception as e:
        logger.error(f"Error in get_rag_response: {str(e)}", exc_info=True)
        return {
            "answer": "I couldn't process your request. Please try again later.",
            "sources": []
        }