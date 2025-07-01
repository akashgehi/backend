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
import random

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1200,
    chunk_overlap=300,
    length_function=len,
    add_start_index=True,
    separators=["\n\n\n", "\n\n", "\n", ". ", "! ", "? ", " ", ""]
)

def verify_embeddings(vector_store):
    if not hasattr(vector_store.db, 'index'):
        raise ValueError("Vector store has no index - check your embedding setup")
    if vector_store.db.index.ntotal == 0:
        raise ValueError("No documents are loaded in the vector store")
    logger.info(f"Vector store contains {vector_store.db.index.ntotal} chunks")

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

def format_chat_history(history: List[Dict[str, str]]) -> str:
    if not history:
        return "No previous conversation context available."
    formatted = []
    for msg in history[-5:]:
        role = "User" if msg["role"] == "user" else "Assistant"
        formatted.append(f"**{role}**: {msg['content']}")
    return "\n\n".join(formatted)

def post_process_answer(answer: str) -> str:
    answer = re.sub(r'\*\*(Empathy|Education|Remedies|Safety Note|Call-to-Action|Response|Note)\*\*', '', answer)
    sections = answer.strip().split('\n\n')
    formatted_sections = []
    for section in sections:
        sentences = [s.strip() for s in section.split('. ') if s.strip()]
        if sentences and '-' not in section:
            formatted_section = '\n'.join(sentences[:3])
            formatted_sections.append(formatted_section)
        elif '-' in section:
            formatted_sections.append(section.strip())
    answer = '\n\n'.join(formatted_sections)
    answer = re.sub(r'^\s*[-*]\s+', '- ', answer, flags=re.MULTILINE)
    return answer.strip()

def get_rag_response(question: str, vector_store, chat_history: List[Dict[str, str]] = []) -> Dict[str, Any]:
    try:
        verify_embeddings(vector_store)
        
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0.3,
            max_output_tokens=650,
            timeout=30
        )

        logger.info(f"Testing retrieval for question: {question}")
        test_docs = vector_store.db.similarity_search_with_score(question, k=4)
        if not test_docs:
            test_docs = vector_store.db.similarity_search_with_score(question, k=4)
        logger.info(f"Found {len(test_docs)} documents")

        retriever = vector_store.db.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": 5, "score_threshold": 0.4}
        )

        prompt = ChatPromptTemplate.from_template("""
        **Role**: You are *Dr. AyurAI* — a warm, helpful Ayurvedic assistant designed to guide users on natural remedies, lifestyle practices, and holistic well-being. You’re part of a professional Ayurvedic service that may also connect users with real doctors, when appropriate.

        **Tone & Personality**: Speak like a kind, wise human — warm, informative, humble, and never generic. Always be empathetic and gently actionable. Never robotic and assume you have already introduced yourself.

        ---

        **Your Core Behaviors**:

        1. **Understand the User**  
        - Be context-aware. Use chat history to follow up naturally or pick up where the user left off.  
        - If the conversation starts fresh, greet politely and briefly share how you can assist.  
        - Reference earlier concerns when appropriate: “As you mentioned earlier…”  

        2. **Use the Right Knowledge**  
        - Use the provided document context when available.  
        - If the context is insufficient, use Ayurvedic knowledge as your foundation — *without saying so directly*.  
        - Never use modern medicine, diagnoses, or prescriptions. Never guess.  

        3. **How You Should Think**  
        - Clarify the user’s concern or intent in your mind.  
        - Explain the Ayurvedic view (briefly).  
        - Offer clear, specific tips or lifestyle remedies.  
        - Encourage conversation, not just one-off answers.  

        ---

        **Response Format (Don't mention these labels the headings which are in ** ** are for you to structure, format the answer wisely in bullets and paragraphs):**

        *always use ticks and crosses for do's and dont's, and fut bullet points.*                                          
        *use emojis as a professional for simple emotions and  dont overdo the emojis.*

        *1 short sentence acknowledging the concern subtly*  
        e.g. “That’s something many people experience, especially during seasonal changes.”

        *2–3 short sentences explaining the Ayurvedic view of the issue.*

        *Give 2–3 relevant tips in bullet points. Each should be concise but actionable.*  
        - Avoid heavy explanations, but include quick rationales.  
        - Mention herbs, routines, or diet tweaks if appropriate.

        *Use italics for a 1-line reminder that every body is unique and a practitioner’s advice may be needed, but give this only when suggesting some serious remedies*

        *End gently, inviting the user to continue, without pressure, while suggesting the next relevant question.*  
        e.g. “Let me know if you’d like a morning routine for your dosha.”

        ---

        **Additional Rules**:

        - Use **short paragraphs** (2–3 sentences each).  
        - Use **double newlines** between sections.  
        - Use *italics* and *bold* where needed for subtlety.  
        - Never start replies with “As an AI...”  
        - Never mention source reliability or fallback models.
        - Always make sure to give a complete answer curating it with the context knowledge and your knowledge, making it sound reliable and professional, yet personal.

        ---

        **When Following Up**:
        - Don’t greet again.
        - Refer back casually: “About your earlier question on skin dryness…” or “Building on what you said about digestion…”

        ---

        **Chat History**:  
        {chat_history}

        **Relevant Document Context**:  
        {context}

        **User’s Question**:  
        {question}

        **Dr. AyurAI’s Response**:

        Provide your response first. After that, suggest 3 natural follow-up questions a user might ask next, as a plain Python list.

        ---
        Final Output Format:

        **Answer**:
        <your full response here>

        **Suggestions**:( dont include this in the response message)
        ["<question1>", "<question2>", "<question3>"]
        """)


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

        raw_output = chain.invoke(question)

        try:
            if "**Suggestions**:" in raw_output:
                response_part, suggestions_part = raw_output.split("**Suggestions**:")
                clean_answer = post_process_answer(response_part.replace("**Answer**:", "").strip())
                suggested_prompts = eval(suggestions_part.strip())
            else:
                clean_answer = post_process_answer(raw_output)
                suggested_prompts = []
        except Exception as e:
            logger.warning(f"Suggestion parsing failed: {str(e)}")
            clean_answer = post_process_answer(raw_output)
            suggested_prompts = []

        return {
            "answer": clean_answer[:2000],
            "suggested_prompts": suggested_prompts
        }

    except Exception as e:
        logger.error(f"Error in get_rag_response: {str(e)}")
        return {
            "answer": f"*System encountered an issue: {str(e)}.*\n\nPlease try again.",
            "suggested_prompts": []
        }

    prompt = ChatPromptTemplate.from_template("""
    Given the recent Ayurvedic chat between user and assistant, suggest 3 thoughtful follow-up questions the user might naturally ask next. Base these on the topic, tone, and assistant’s suggestions.

    Chat history:
    {chat_history}

    Respond ONLY as a Python list of 3 questions, e.g., ["...", "...", "..."]
    """)

    chain = (
        RunnablePassthrough()
        | (lambda input: {"chat_history": format_chat_history(input)})
        | prompt
        | ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=os.getenv("GOOGLE_API_KEY"))
        | StrOutputParser()
    )

    response = chain.invoke(chat_history)
    try:
        return eval(response)
    except Exception:
        return ["Let’s explore more about Ayurvedic routines.", "What else can I do for balance?", "Any dietary suggestions?"]