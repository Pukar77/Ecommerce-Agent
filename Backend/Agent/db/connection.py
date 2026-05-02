import os
from supabase import create_client, Client
from dotenv import load_dotenv

# Load env from the Agent/db/.env file
# os.path.abspath handles the case where __file__ is just a filename (no directory part)
_env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
load_dotenv(dotenv_path=_env_path)


def get_supabase_client() -> Client:
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")

    if not url or not key:
        raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set in Agent/db/.env")

    return create_client(url, key)
