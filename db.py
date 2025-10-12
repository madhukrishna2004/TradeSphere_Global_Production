import psycopg2
import os
from dotenv import load_dotenv

load_dotenv()

def get_db():
    return psycopg2.connect(os.getenv("DATABASE_URL"))
