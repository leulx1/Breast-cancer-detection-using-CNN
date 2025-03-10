import os
import psycopg2

DB_HOST = "localhost"
DB_NAME = "BCDB"
DB_USER = "postgres"
DB_PASSWORD = "1234"
DB_PORT = "5432"

def get_db_connection():
    conn = psycopg2.connect(
        host=DB_HOST,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        port=DB_PORT
    )
    return conn