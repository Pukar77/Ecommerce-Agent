"""
Unified entry point — RAG + Agentic Database tools.

Routing logic:
  - Buy / purchase intent  → tool-calling agent (buy_product via Supabase)
  - Stock / availability   → tool-calling agent (check_stock via Supabase)
  - Everything else        → RAG chain (product info from Qdrant)
    - If a specific product number is mentioned, fetch 30 candidates and
      filter to chunks that contain the exact "Product N:" text, so
      similar-numbered products (e.g. 17, 71) don't pollute the context.
"""

import sys
import os
import re

# ── path setup ────────────────────────────────────────────────────────────────
_rag_dir   = os.path.dirname(os.path.abspath(__file__))
_agent_dir = os.path.join(_rag_dir, "..", "Agent")
sys.path.insert(0, _rag_dir)
sys.path.insert(0, _agent_dir)

from dotenv import load_dotenv
load_dotenv(dotenv_path=os.path.join(_rag_dir, ".env"))
load_dotenv(dotenv_path=os.path.join(_agent_dir, "db", ".env"))

# ── imports ───────────────────────────────────────────────────────────────────
from retriver_03 import get_retrevial, get_vectorstore
from rag_chain_04 import get_rag_chain, get_llm, format_docs
from document_loader_text_splitter_01 import load_pdf, chunking
from vector_store_02 import vector_stores
from qdrant_client import QdrantClient
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from tools.database_tools import check_stock, buy_product


# ── intent detection ──────────────────────────────────────────────────────────

BUY_PATTERNS = re.compile(
    r"\b(buy|purchase|order|get me|i want to buy|i'd like to buy|place order)\b",
    re.IGNORECASE,
)
STOCK_PATTERNS = re.compile(
    r"\b(stock|availability|available|how many|units left|in stock|check stock)\b",
    re.IGNORECASE,
)


def detect_intent(text: str) -> str:
    """Returns 'buy', 'stock', or 'rag'."""
    if BUY_PATTERNS.search(text):
        return "buy"
    if STOCK_PATTERNS.search(text):
        return "stock"
    return "rag"


def extract_product_id(text: str) -> int | None:
    """Extract the first number from the user message as product_id."""
    match = re.search(r"\b(\d+)\b", text)
    return int(match.group(1)) if match else None


# ── Qdrant helpers ────────────────────────────────────────────────────────────

def collection_exists():
    client = QdrantClient(url="http://localhost:6333")
    existing = [c.name for c in client.get_collections().collections]
    return "products_agentic_ai_new" in existing


def ingest():
    pdf_path = os.path.join(_rag_dir, "ecommerce_products_rag.pdf")
    print(f"📄 Loading PDF: {pdf_path}")
    documents = load_pdf(pdf_path)
    chunks = chunking(documents)
    print(f"✂️  Total chunks: {len(chunks)}")
    vector_stores(chunks)
    print("✅ Ingestion complete!")


# ── Smart RAG query ───────────────────────────────────────────────────────────

def rag_answer(question: str, vectorstore, llm) -> str:
    """
    Answer a product question using RAG.
    If a specific product number is mentioned, fetch 30 candidates and
    filter to only chunks containing 'Product N:' to avoid false matches.
    """
    pid = extract_product_id(question)

    if pid is not None:
        # Fetch wide set of candidates
        wide_retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 30},
        )
        all_docs = wide_retriever.invoke(f"Smart Item {pid} product {pid}")

        # Keep only chunks that explicitly mention this exact product
        exact_marker = f"Product {pid}:"
        matched = [d for d in all_docs if exact_marker in d.page_content]

        if not matched:
            # Fallback: try any chunk mentioning the product name
            matched = [d for d in all_docs if f"Smart Item {pid}" in d.page_content]

        context = "\n\n".join(d.page_content for d in matched[:4]) if matched else "No information found."
    else:
        # General question — use normal retriever
        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})
        docs      = retriever.invoke(question)
        context   = format_docs(docs)

    prompt = ChatPromptTemplate.from_template("""
You are a helpful ecommerce product assistant.
Use the following retrieved context to answer the user's question accurately.
If the answer is not in the context, say "I don't have information about that product."

Context:
{context}

Question:
{question}

Answer:
""")

    chain  = prompt | llm | StrOutputParser()
    return chain.invoke({"context": context, "question": question})


# ── Database agent (buy / stock only) ────────────────────────────────────────

DB_SYSTEM_PROMPT = """You are an ecommerce assistant that handles purchases and stock checks.
You have two tools:
1. buy_product   — purchase a product (decreases stock by 1 in the database)
2. check_stock   — check how many units of a product are available

Rules:
- Always call the appropriate tool immediately — do not ask for confirmation.
- Product IDs are integers (e.g. "product 7" → product_id=7).
- After a purchase, confirm what was bought and the remaining stock.
- After a stock check, clearly state the available quantity."""


def run_db_agent(llm_with_tools, tools_by_name: dict, user_input: str) -> str:
    """Tool-calling loop for buy/stock operations only."""
    messages = [
        SystemMessage(content=DB_SYSTEM_PROMPT),
        HumanMessage(content=user_input),
    ]

    for _ in range(5):
        response = llm_with_tools.invoke(messages)
        messages.append(response)

        if not response.tool_calls:
            return response.content

        for tc in response.tool_calls:
            tool_name = tc["name"]
            tool_args = tc["args"]
            tool_id   = tc["id"]

            print(f"\n🔧 Calling tool: {tool_name}({tool_args})")
            result = tools_by_name[tool_name].invoke(tool_args)
            print(f"   Result: {result}")

            messages.append(ToolMessage(content=str(result), tool_call_id=tool_id))

    return response.content


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("🚀 Initializing RAG + Agent pipeline...")

    if not collection_exists():
        print("⚠️  Collection not found. Running ingestion first...")
        ingest()

    # Shared components
    vectorstore = get_vectorstore()
    llm         = get_llm()

    # DB tools (buy / stock)
    db_tools      = [check_stock, buy_product]
    tools_by_name = {t.name: t for t in db_tools}
    llm_with_tools = llm.bind_tools(db_tools)

    print("✅ Pipeline ready!")
    print("   • Product questions → RAG (Qdrant)")
    print("   • Buy / stock       → Agent (Supabase)")
    print("💬 Type 'exit' to quit\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n👋 Goodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit"):
            print("👋 Goodbye!")
            break

        print()
        intent = detect_intent(user_input)

        if intent in ("buy", "stock"):
            answer = run_db_agent(llm_with_tools, tools_by_name, user_input)
        else:
            answer = rag_answer(user_input, vectorstore, llm)

        print(f"\n🤖 Assistant: {answer}")
        print("\n" + "=" * 50 + "\n")


if __name__ == "__main__":
    main()
