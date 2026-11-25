from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from src.api.models import Base

import os
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./data/financial_stability.db")
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)
connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db():
    Base.metadata.create_all(bind=engine)
    if DATABASE_URL.startswith("sqlite"):
        with engine.begin() as connection:
            columns = {
                row[1]: row
                for row in connection.exec_driver_sql("PRAGMA table_info(users)").all()
            }
            if "lifestyle" not in columns:
                connection.exec_driver_sql(
                    "ALTER TABLE users ADD COLUMN lifestyle VARCHAR DEFAULT 'student'"
                )
            if "next_income_date" not in columns:
                connection.exec_driver_sql(
                    "ALTER TABLE users ADD COLUMN next_income_date DATETIME"
                )
            if "emergency_reserve" not in columns:
                connection.exec_driver_sql(
                    "ALTER TABLE users ADD COLUMN emergency_reserve FLOAT DEFAULT 0"
                )

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
