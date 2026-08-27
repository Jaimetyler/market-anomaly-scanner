import os

from dotenv import load_dotenv


load_dotenv()

MASSIVE_API_KEY = os.getenv("MASSIVE_API_KEY")

if not MASSIVE_API_KEY:
    raise RuntimeError(
        "MASSIVE_API_KEY is missing. Add it to your .env file."
    )