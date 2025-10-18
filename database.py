import sqlite3
import json
from datetime import datetime

DB_FILE = "chatbot_memory.db"


def get_connection():
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS learned_intents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            pattern TEXT,
            response TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)


    cursor.execute("""
        CREATE TABLE IF NOT EXISTS memory (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT UNIQUE,
            name TEXT,
            mood TEXT,
            last_intent TEXT,
            history TEXT,
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.commit()
    conn.close()


def save_user_context(user_id, context):
    """Insert or update user memory."""
    conn = get_connection()
    cursor = conn.cursor()

    data = {
        "name": context.get("name"),
        "mood": context.get("mood"),
        "last_intent": context.get("last_intent"),
        "history": json.dumps(context.get("history", [])),
    }

    cursor.execute("""
        INSERT INTO memory (user_id, name, mood, last_intent, history, last_updated)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(user_id) DO UPDATE SET
            name = excluded.name,
            mood = excluded.mood,
            last_intent = excluded.last_intent,
            history = excluded.history,
            last_updated = excluded.last_updated
    """, (user_id, data["name"], data["mood"], data["last_intent"], data["history"], datetime.now()))

    conn.commit()
    conn.close()


def load_user_context(user_id):
    """Fetch memory for a specific user."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM memory WHERE user_id = ?", (user_id,))
    row = cursor.fetchone()
    conn.close()

    if row:
        return {
            "name": row["name"],
            "mood": row["mood"],
            "last_intent": row["last_intent"],
            "history": json.loads(row["history"]) if row["history"] else []
        }
    return None


def load_all_memory():
    """Load all user memories on startup."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM memory")
    rows = cursor.fetchall()
    conn.close()

    memory = {}
    for row in rows:
        memory[row["user_id"]] = {
            "name": row["name"],
            "mood": row["mood"],
            "last_intent": row["last_intent"],
            "history": json.loads(row["history"]) if row["history"] else []
        }
    return memory

def store_learned_intent(user_id, pattern, response):
    """Store a newly learned user pattern + response."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO learned_intents (user_id, pattern, response)
        VALUES (?, ?, ?)
    """, (user_id, pattern, response))
    conn.commit()
    conn.close()


def get_all_learned_intents():
    """Fetch all learned patterns and responses."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT pattern, response FROM learned_intents")
    rows = cursor.fetchall()
    conn.close()
    return [{"pattern": row["pattern"], "response": row["response"]} for row in rows]

def get_learned_count():
    """Return how many learned responses exist in the DB."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM learned_intents")
    count = cursor.fetchone()[0]
    conn.close()
    return count
