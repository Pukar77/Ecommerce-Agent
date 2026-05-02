"""
Run this script ONCE to seed the Supabase `products` table with 50 products,
each starting with stock = 2.

Table schema (create in Supabase SQL editor before running):

    CREATE TABLE products (
        id          SERIAL PRIMARY KEY,
        product_id  INT UNIQUE NOT NULL,
        name        TEXT NOT NULL,
        stock       INT NOT NULL DEFAULT 2
    );
"""

import sys
import os

# Allow running from any directory
sys.path.insert(0, os.path.dirname(__file__))

from connection import get_supabase_client

PRODUCTS = [{"product_id": i, "name": f"Smart Item {i}", "stock": 2} for i in range(1, 51)]


def seed():
    client = get_supabase_client()

    print("🌱 Seeding products table...")
    response = client.table("products").upsert(PRODUCTS, on_conflict="product_id").execute()

    if response.data:
        print(f"✅ Seeded {len(response.data)} products successfully.")
    else:
        print("⚠️  Seed may have failed or table already has data.")
        print(response)


if __name__ == "__main__":
    seed()
