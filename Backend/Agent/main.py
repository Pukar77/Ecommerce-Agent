"""
Entry point for the Agentic RAG ecommerce assistant.

Run from the Backend/Agent directory:
    python main.py
"""

import sys
import os

# Ensure local modules resolve correctly
sys.path.insert(0, os.path.dirname(__file__))

from agent import build_agent


def main():
    print("🚀 Starting Ecommerce Agent...")
    print("   - RAG: Qdrant (product catalog)")
    print("   - DB : Supabase (stock management)")
    print("💬 Type 'exit' to quit\n")

    executor = build_agent()
    chat_history = []

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
        response = executor.invoke({
            "input": user_input,
            "chat_history": chat_history,
        })

        answer = response.get("output", "")
        print(f"\n🤖 Agent: {answer}")
        print("\n" + "=" * 60 + "\n")

        # Keep last 10 turns in memory
        from langchain_core.messages import HumanMessage, AIMessage
        chat_history.append(HumanMessage(content=user_input))
        chat_history.append(AIMessage(content=answer))
        if len(chat_history) > 20:
            chat_history = chat_history[-20:]


if __name__ == "__main__":
    main()
