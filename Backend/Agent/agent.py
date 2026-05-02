"""
Agentic RAG — combines the existing RAG retriever with Supabase purchase tools.

The agent can:
  - Answer product questions using the RAG knowledge base (Qdrant)
  - Check stock levels via Supabase
  - Purchase products (decrease stock by 1) via Supabase
"""

import sys
import os

# Make Rag modules importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "Rag"))

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.tools import tool
from langchain_qdrant import QdrantVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from dotenv import load_dotenv

from tools.database_tools import check_stock, buy_product

# Load env files — use abspath so __file__ resolves correctly from any cwd
_agent_dir = os.path.dirname(os.path.abspath(__file__))
load_dotenv(dotenv_path=os.path.join(_agent_dir, "..", "Rag", ".env"))
load_dotenv(dotenv_path=os.path.join(_agent_dir, "db", ".env"))


# ── RAG retriever as a LangChain tool ────────────────────────────────────────

def _build_rag_tool():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    client = QdrantClient(url="http://localhost:6333")
    vectorstore = QdrantVectorStore(
        client=client,
        collection_name="products_agentic_ai_new",
        embedding=embeddings,
        content_payload_key="page_content",
    )
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})

    @tool
    def search_products(query: str) -> str:
        """
        Search the product catalog using semantic search.
        Use this to answer questions about product details, prices, descriptions,
        categories, variants, shelf life, and general product information.

        Args:
            query: A natural language question or search phrase about products.

        Returns:
            Relevant product information from the catalog.
        """
        docs = retriever.invoke(query)
        if not docs:
            return "No relevant product information found."
        return "\n\n".join(doc.page_content for doc in docs)

    return search_products


# ── Agent builder ─────────────────────────────────────────────────────────────

def build_agent() -> AgentExecutor:
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0,
    )

    search_products = _build_rag_tool()
    tools = [search_products, check_stock, buy_product]

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are a helpful ecommerce assistant with access to a product catalog and inventory system.

You have three tools available:
1. search_products  — search the product catalog for information (prices, descriptions, etc.)
2. check_stock      — check how many units of a product are available
3. buy_product      — purchase a product (decreases stock by 1 in the database)

Guidelines:
- When a user wants to BUY a product, ALWAYS call buy_product directly with the product ID.
  Do NOT just check stock — actually complete the purchase.
- When a user asks about product details, use search_products.
- When a user asks about stock/availability without buying, use check_stock.
- Product IDs are numbers (e.g. "Product 8" → product_id = 8).
- Be concise and friendly in your responses.
- After a successful purchase, confirm what was bought and the remaining stock.""",
        ),
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True)
