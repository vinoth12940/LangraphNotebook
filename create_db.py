import sqlite3
import requests
import os

def create_chinook_db():
    """Download and create the Chinook database."""
    # URL of the Chinook database SQL script
    url = "https://raw.githubusercontent.com/lerocha/chinook-database/master/ChinookDatabase/DataSources/Chinook_Sqlite.sql"
    
    # Download the SQL script
    print("Downloading Chinook database SQL script...")
    response = requests.get(url)
    sql_script = response.text
    
    # Create a new SQLite database file
    db_path = "chinook.db"
    
    # Remove existing database if it exists
    if os.path.exists(db_path):
        os.remove(db_path)
    
    print(f"Creating database at: {os.path.abspath(db_path)}")
    
    # Create new database and execute the script
    connection = sqlite3.connect(db_path)
    connection.executescript(sql_script)
    connection.close()
    
    print("Database created successfully!")
    return db_path

if __name__ == "__main__":
    create_chinook_db() 