from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from typing import List, Dict, Any
import os
import re
import logging
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 1. Verify embeddings are working
def verify_embeddings(vector_store):
    if not hasattr(vector_store.db, 'index'):
        raise ValueError("Vector store has no index - check your embedding setup")
    if vector_store.db.index.ntotal == 0:
        raise ValueError("No documents are loaded in the vector store")
    logger.info(f"Vector store contains {vector_store.db.index.ntotal} chunks")

# 2. Optimized text splitter with better PDF handling
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1200,
    chunk_overlap=300,
    length_function=len,
    add_start_index=True,
    separators=["\n\n\n", "\n\n", "\n", ". ", "! ", "? ", " ", ""]
)

# 3. Enhanced document formatting
def format_docs(docs: List[Document]) -> str:
    if not docs:
        return "[No matching documents found]"
    formatted = []
    for i, doc in enumerate(docs):
        content = re.sub(r'\s+', ' ', doc.page_content)
        content = re.sub(r'-\n', '', content)
        formatted.append(
            f"**Document {i+1}** (Source: {doc.metadata.get('source', '?')}, Page: {doc.metadata.get('page', '?')}):\n"
            f"{content[:800]}{'...' if len(content) > 800 else ''}"
        )
    return "\n\n".join(formatted)

# 4. Format chat history for prompt
def format_chat_history(history: List[Dict[str, str]]) -> str:
    if not history:
        return "No previous conversation context available."
    formatted = []
    for msg in history[-5:]:  # Limit to last 5 messages for relevance
        role = "User" if msg["role"] == "user" else "Assistant"
        formatted.append(f"**{role}**: {msg['content']}")
    return "\n\n".join(formatted)

# 5. Post-process response to enforce formatting
def post_process_answer(answer: str) -> str:
    # Remove any unintended labels
    answer = re.sub(r'\*\*(Empathy|Education|Remedies|Safety Note|Call-to-Action|Response|Note)\*\*', '', answer)
    # Ensure double newlines between sections
    sections = answer.strip().split('\n\n')
    formatted_sections = []
    for section in sections:
        # Split into sentences and ensure single newlines within paragraphs
        sentences = [s.strip() for s in section.split('. ') if s.strip()]
        if sentences and '-' not in section:  # Non-list sections
            formatted_section = '\n'.join(sentences[:3])  # Limit to 3 sentences
            formatted_sections.append(formatted_section)
        elif '-' in section:  # List sections
            formatted_sections.append(section.strip())
    answer = '\n\n'.join(formatted_sections)
    # Ensure bullet points are properly formatted
    answer = re.sub(r'^\s*[-*]\s+', '- ', answer, flags=re.MULTILINE)
    return answer.strip()

# 6. Main RAG function with enhanced formatting
def get_rag_response(question: str, vector_store, chat_history: List[Dict[str, str]] = []) -> Dict[str, Any]:
    try:
        # Step 1: Verify basic setup
        verify_embeddings(vector_store)
        
        # Step 2: Initialize LLM with timeout
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0.3,  # Increased for better formatting adherence
            max_output_tokens=650,
            timeout=30
        )

        # Step 3: Test retrieval with similarity scores
        logger.info(f"Testing retrieval for question: {question}")
        test_docs = vector_store.db.similarity_search_with_score(question, k=4)
        if not test_docs:
            logger.warning("No documents retrieved with similarity search. Falling back to default search.")
            test_docs = vector_store.db.similarity_search_with_score(question, k=4)
        logger.info(f"Found {len(test_docs)} documents")
        for i, (doc, score) in enumerate(test_docs):
            logger.info(f"Doc {i+1}: {doc.metadata.get('source', '?')} pg. {doc.metadata.get('page', '?')} (Score: {score:.2f})")
            logger.debug(f"Content: {doc.page_content[:200]}...")

        # Step 4: Configure retriever with adjusted threshold
        retriever = vector_store.db.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": 5, "score_threshold": 0.4}
        )
        prompt = ChatPromptTemplate.from_template("""
        **Role**: You are Dr. AyurAI, an expert Ayurvedic assistant with a warm, conversational tone. Your goal is to provide accurate, evidence-based remedies from the knowledge base, supplemented by general Ayurvedic principles, in a natural, ongoing chat flow that’s highly readable and engaging.

        **Instructions**:
        1. **Context Awareness**: Use chat history to maintain conversation flow. Reference prior messages naturally (e.g., "Following your question..." or "You mentioned earlier..."). Avoid repetitive greetings like "Namaste" or formal introductions.
        2. **Chain-of-Thought**:
           - Identify the core issue in the question.
           - Match it with relevant document content or prior conversation.
           - Craft a concise, structured, and engaging response.
        3. **Response Structure** (labels for internal guidance only, do not include in output):
           - **Empathy**: 1 sentence acknowledging the concern for new topics, subtle and not overly emotional (e.g., "That can be tough...").
           - **Education**: 2-3 sentences explaining the Ayurvedic perspective in a single paragraph.
           - **Remedies**: 1-2 remedies in bullet points, each with 1-2 sentence explanations.
           - **Safety Note**: One-sentence disclaimer in *italics*.
           - **Call-to-Action**: One sentence with an interactive question or suggestion (e.g., "Anything else you’d like to explore?").
        4. **Formatting Rules**:
           - Do NOT include labels (**Empathy**, **Education**, etc.) in the response; use them only to structure the content internally.
           - Keep paragraphs short (2-3 sentences, max 50 words) with single newlines (\n) between sentences within a section.
           - Use double newlines (\n\n) to separate sections (empathy, education, remedies, etc.).
           - Use bullet points (- or *) for remedies or lists, with 1-2 sentences per item.
           - Use *italics* for the safety note and emphasis.
           - Ensure consistent spacing and avoid clutter.
        5. **Follow-Up Flow**: For follow-ups, start directly with the answer, linking to prior context if relevant (e.g., "Following your insomnia question...").
        6. **Fallback**: If no documents are found, start with a note in *italics* with general knowledge and mention the limitation.
        7. **Example Response** (for formatting guidance):
           That can be tough.

           Issue is linked to Vata.
           Balancing it helps.
           Warm foods aid relaxation.

           - Tip 1: Description one.
           Description two.
           - Tip 2: Description one.
           Description two.

           *Consult a professional.*

           Anything else you’d like to explore?

        **Chat History**:
        {chat_history}

        **Retrieved Documents**:
        {context}

        **User Question**:
        {question}

        **Dr. AyurAI's Response**:
        """)

        # Step 6: Build the chain
        chain = (
            {
                "context": retriever | format_docs,
                "question": RunnablePassthrough(),
                "chat_history": lambda x: format_chat_history(chat_history)
            }
            | prompt
            | llm
            | StrOutputParser()
        )

        # Step 7: Execute with fallback
        answer = chain.invoke(question)
        
        # Step 8: Post-process answer
        answer = post_process_answer(answer)
        if not test_docs or all(score < 0.4 for _, score in test_docs):
            logger.warning("Low relevance documents. Using fallback response.")
            answer = f"*Limited information found in documents.*\n\n{answer}"

        # Step 9: Calculate usage information
        input_tokens = (len(question) + sum(len(msg['content']) for msg in chat_history[-5:])) // 4
        output_tokens = len(answer) // 4
        referenced_files = list(set(doc.metadata.get("source", "Unknown") for doc, _ in test_docs))

        # Convert numpy.float32 scores to Python float
        sources = [
            {
                "source": doc.metadata.get("source", "Unknown"),
                "page": doc.metadata.get("page", "N/A"),
                "score": float(score)
            }
            for doc, score in test_docs[:5]
        ]

        return {
            "answer": answer[:2000],
            "sources": sources,
            "usage": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
                "files": referenced_files
            }
        }

    except Exception as e:
        logger.error(f"Error in get_rag_response: {str(e)}")
        return {
            "answer": f"*System encountered an issue: {str(e)}.*\n\nPlease try again.",
            "sources": [],
            "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "files": []}
        }