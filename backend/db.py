# backend/db.py
import os
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

# Load environment variables from .env file

load_dotenv(dotenv_path=Path(__file__).parent / ".env")

DATABASE_URL = os.getenv("DATABASE_URL")  # set in .env

if not DATABASE_URL:
    raise RuntimeError(
        "DATABASE_URL is not set. Create backend/.env or export the env var. "
        "Example: postgresql://drl_user:YOUR_PASS@localhost:5432/drl_cache"
    )

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
