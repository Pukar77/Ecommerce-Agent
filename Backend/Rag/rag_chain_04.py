from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv

load_dotenv()


def get_llm():
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.2
    )
    return llm


def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])


def get_rag_chain(retriever):

    llm = get_llm()

    prompt = ChatPromptTemplate.from_template("""
    You are a helpful ecommerce product assistant.
    Use the following retrieved context to answer the user's question accurately.
    If the answer is not in the context, say "I don't have enough information to answer that."

    Context:
    {context}

    Question:
    {question}

    Answer:
    """)

    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain