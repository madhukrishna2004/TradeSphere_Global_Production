import psycopg2
from psycopg2 import sql

# Database connection URL
DATABASE_URL = "postgresql://central_db_gpb1_user:9SiZ5xmQi6OB8NOl6HvK6XAjrXvEO62F@dpg-cv59gt8gph6c73apj7qg-a.singapore-postgres.render.com/central_db_gpb1"

# Connect to the PostgreSQL database
try:
    conn = psycopg2.connect(DATABASE_URL)
    cursor = conn.cursor()

    # SQL to create the supplier_demand table
    create_table_query = '''
    CREATE TABLE IF NOT EXISTS supplier_demand (
        id SERIAL PRIMARY KEY,
        username TEXT NOT NULL,
        password TEXT NOT NULL,
        consent TEXT,  -- URL or file name for PDF/DOC
        commodity_link TEXT
    );
    '''

    # Execute the query
    cursor.execute(create_table_query)
    conn.commit()

    print("✅ Table 'supplier_demand' created successfully.")

except Exception as e:
    print("❌ Error occurred:", e)

finally:
    if cursor:
        cursor.close()
    if conn:
        conn.close()
