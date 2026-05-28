from dotenv import load_dotenv
import os
import psycopg2

load_dotenv()

try:
    conn = psycopg2.connect(
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT"),
        database=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD")
    )

    print("Conexión exitosa")

    cur = conn.cursor()

    cur.execute("SELECT current_database();")

    result = cur.fetchone()
    print("Base de datos conectada:", result[0])

    cur.close()
    conn.close()

except Exception as e:
    print("Error:")
    print(e)
