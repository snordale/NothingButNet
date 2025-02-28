#!/usr/bin/env python3
import logging
from nothingbutnet.database import init_db, Base
from sqlalchemy import create_engine, text
import os
from dotenv import load_dotenv

def update_schema():
    """Update database schema with new columns"""
    logging.info("Updating database schema...")
    
    # Load database URL from environment
    load_dotenv()
    DATABASE_URL = os.getenv('NBA_DATABASE_URL', 'postgresql://localhost/nothingbutnet')
    
    # Create engine
    engine = create_engine(DATABASE_URL)
    
    try:
        # Add new columns if they don't exist
        with engine.connect() as conn:
            for column in [
                ('position', 'INTEGER'),
                ('games_behind', 'FLOAT'),
                ('conference', 'VARCHAR'),
                ('win_pct', 'FLOAT'),
                ('points_per_game', 'FLOAT'),
                ('points_allowed_per_game', 'FLOAT')
            ]:
                try:
                    conn.execute(text(
                        f"ALTER TABLE team_stats ADD COLUMN IF NOT EXISTS {column[0]} {column[1]}"
                    ))
                    conn.commit()
                    logging.info(f"Added or verified column {column[0]}")
                except Exception as e:
                    logging.warning(f"Error adding column {column[0]}: {e}")
        
        logging.info("Schema update complete")
        
    except Exception as e:
        logging.error(f"Error updating schema: {e}")
        raise

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    update_schema() 