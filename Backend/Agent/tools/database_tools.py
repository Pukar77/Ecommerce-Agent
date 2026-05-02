"""
LangChain tools for interacting with the Supabase products table.
"""

import sys
import os

# Ensure db module is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from langchain_core.tools import tool
from db.connection import get_supabase_client


@tool
def check_stock(product_id: int) -> str:
    """
    Check the current stock level of a product by its product ID number.
    Use this before attempting a purchase to verify availability.

    Args:
        product_id: The numeric ID of the product (e.g. 8 for Product 8).

    Returns:
        A message with the current stock level, or an error message.
    """
    try:
        client = get_supabase_client()
        response = (
            client.table("products")
            .select("product_id, name, stock")
            .eq("product_id", product_id)
            .single()
            .execute()
        )

        if not response.data:
            return f"❌ Product {product_id} not found in the database."

        product = response.data
        return (
            f"📦 Product {product['product_id']} ({product['name']}) "
            f"has {product['stock']} unit(s) in stock."
        )

    except Exception as e:
        return f"❌ Error checking stock for product {product_id}: {str(e)}"


@tool
def buy_product(product_id: int) -> str:
    """
    Purchase a product by decreasing its stock by 1 in the database.
    Always check stock first. If stock is 0, the purchase will be refused.

    Args:
        product_id: The numeric ID of the product to purchase (e.g. 8 for Product 8).

    Returns:
        A success or failure message.
    """
    try:
        client = get_supabase_client()

        # Fetch current stock
        response = (
            client.table("products")
            .select("product_id, name, stock")
            .eq("product_id", product_id)
            .single()
            .execute()
        )

        if not response.data:
            return f"❌ Product {product_id} not found in the database."

        product = response.data
        current_stock = product["stock"]

        if current_stock <= 0:
            return (
                f"❌ Sorry, Product {product_id} ({product['name']}) is out of stock. "
                f"Cannot complete the purchase."
            )

        # Decrease stock by 1
        new_stock = current_stock - 1
        update_response = (
            client.table("products")
            .update({"stock": new_stock})
            .eq("product_id", product_id)
            .execute()
        )

        if update_response.data:
            return (
                f"✅ Successfully purchased Product {product_id} ({product['name']})! "
                f"Remaining stock: {new_stock} unit(s)."
            )
        else:
            return f"❌ Failed to update stock for Product {product_id}. Please try again."

    except Exception as e:
        return f"❌ Error processing purchase for product {product_id}: {str(e)}"
