# check_db.py
import sqlite3
import os

# --- Define the path to your database ---
# Make sure this path is correct relative to where you run this script
DB_PATH = os.path.join( "db", "ab.nn.db") # Adjust path if needed

def check_database():
    if not os.path.exists(DB_PATH):
        print(f"Database file not found at: {DB_PATH}")
        return

    try:
        # Connect to the SQLite database
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        # --- Query 1: List all tables ---
        print("--- Database Tables ---")
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        if tables:
            for table in tables:
                print(f"  - {table[0]}")
        else:
            print("  No tables found.")

        # --- Query 2: Check contents of a likely table (replace 'nn_results' if different) ---
        # You need to know the actual table name. Let's assume it's related to 'nn' and 'results'.
        # Common names might be 'neural_networks', 'evaluations', 'nn_results', 'models', etc.
        # Let's try a common one or iterate through tables if you're unsure.
        table_name = None
        for t in tables:
             # Guess a table name, e.g., one containing 'nn'
             if 'nn' in t[0].lower() or 'model' in t[0].lower():
                  table_name = t[0]
                  break
        # If no obvious table found, you might need to inspect the first one or check docs/source
        if not table_name and tables:
             table_name = tables[0][0] # Use the first table found
             print(f"\nNo obvious 'nn' table found. Checking first table: {table_name}")

        if table_name:
            print(f"\n--- Contents of table '{table_name}' (first 5 rows) ---")
            cursor.execute(f"SELECT * FROM {table_name} LIMIT 5;")
            rows = cursor.fetchall()

            # Get column names
            column_names = [description[0] for description in cursor.description]
            print("Columns:", column_names)

            if rows:
                for row in rows:
                    # Print row data, potentially truncating long entries like code
                    formatted_row = []
                    for item in row:
                        if isinstance(item, str) and len(item) > 100:
                            formatted_row.append(item[:97] + "...")
                        else:
                            formatted_row.append(item)
                    print(formatted_row)
            else:
                print("  Table is empty or query returned no results.")

            # --- Query 3: Count total entries ---
            cursor.execute(f"SELECT COUNT(*) FROM {table_name};")
            count = cursor.fetchone()[0]
            print(f"\n--- Total entries in '{table_name}': {count} ---")

        else:
            print("\nNo table found to inspect.")

    except sqlite3.Error as e:
        print(f"An error occurred while accessing the database: {e}")
    finally:
        if conn:
            conn.close()
            print("\nDatabase connection closed.")


if __name__ == "__main__":
    check_database()
