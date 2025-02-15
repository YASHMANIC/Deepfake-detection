import sqlite3
import os

def view_database_contents():
    db_path = 'instance/app.db'
    if not os.path.exists(db_path):
        print(f"Database file not found at {db_path}")
        return
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get list of tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        
        print("\nDatabase Tables:")
        print("-" * 50)
        for table in tables:
            table_name = table[0]
            print(f"\nTable: {table_name}")
            
            # Get table schema
            cursor.execute(f"PRAGMA table_info({table_name});")
            columns = cursor.fetchall()
            print("\nColumns:")
            for col in columns:
                print(f"  {col[1]} ({col[2]})")
            
            # Get table contents
            cursor.execute(f"SELECT * FROM {table_name};")
            rows = cursor.fetchall()
            print(f"\nRecords ({len(rows)}):")
            for row in rows:
                print(f"  {row}")
            print("-" * 50)
        
    except sqlite3.Error as e:
        print(f"SQLite error: {e}")
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    view_database_contents()
