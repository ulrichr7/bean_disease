import os
import sqlite3
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "data"))
DB_PATH = os.path.join(DATA_DIR, "metadata.db")

os.makedirs(DATA_DIR, exist_ok=True)

def get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    return conn

def init_db():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS uploads (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT,
            label TEXT,
            saved_path TEXT,
            uploaded_at TEXT
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS models (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_path TEXT,
            created_at TEXT,
            samples INTEGER
        )
        """
    )
    conn.commit()
    conn.close()

def record_upload(filename: str, label: str, saved_path: str, uploaded_at: datetime = None):
    if uploaded_at is None:
        uploaded_at = datetime.utcnow()
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO uploads (filename, label, saved_path, uploaded_at) VALUES (?, ?, ?, ?)",
        (filename, label, saved_path, uploaded_at.isoformat()),
    )
    conn.commit()
    conn.close()

def record_model_version(model_path: str, samples: int, created_at: datetime = None):
    if created_at is None:
        created_at = datetime.utcnow()
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO models (model_path, created_at, samples) VALUES (?, ?, ?)",
        (model_path, created_at.isoformat(), samples),
    )
    conn.commit()
    conn.close()

def list_models():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT id, model_path, created_at, samples FROM models ORDER BY id DESC")
    rows = cur.fetchall()
    conn.close()
    return rows
