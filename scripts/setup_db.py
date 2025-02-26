#!/usr/bin/env python3
import os
import sys
import subprocess
import logging
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.nothingbutnet.database import init_db

def check_postgres():
    """Check if PostgreSQL is installed and running"""
    try:
        # Check if PostgreSQL is installed
        subprocess.run(['which', 'psql'], check=True, capture_output=True)
        
        # Check if PostgreSQL service is running
        subprocess.run(['pg_isready'], check=True, capture_output=True)
        
        return True
    except subprocess.CalledProcessError:
        return False

def create_database():
    """Create the database if it doesn't exist"""
    try:
        # Extract database name from URL
        db_url = os.getenv('NBA_DATABASE_URL', 'postgresql://localhost/nothingbutnet')
        db_name = db_url.split('/')[-1]
        
        # Check if database exists
        result = subprocess.run(
            ['psql', '-lqt'],
            capture_output=True,
            text=True
        )
        
        if db_name not in result.stdout:
            print(f"Creating database {db_name}...")
            subprocess.run(['createdb', db_name], check=True)
            print("Database created successfully!")
        else:
            print(f"Database {db_name} already exists.")
        
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error creating database: {e}")
        return False

def main():
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Check PostgreSQL installation
    if not check_postgres():
        print("PostgreSQL is not installed or not running.")
        print("Please install PostgreSQL and ensure the service is running.")
        print("On macOS with Homebrew: brew install postgresql@15")
        print("Then start the service: brew services start postgresql@15")
        sys.exit(1)
    
    # Create database
    if not create_database():
        print("Failed to create database. Please check PostgreSQL configuration.")
        sys.exit(1)
    
    # Initialize database schema
    print("Initializing database schema...")
    try:
        session = init_db(drop_all=False)
        session.close()
        print("Database setup complete!")
    except Exception as e:
        print(f"Error initializing database schema: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 