import sqlite3

conn = sqlite3.connect("chatbot_memory.db")
cursor = conn.cursor()

cursor.execute("SELECT user_id, last_updated FROM memory;")
for row in cursor.fetchall():
    print(row)

conn.close()
