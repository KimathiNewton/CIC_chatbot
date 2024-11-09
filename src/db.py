import psycopg2
from psycopg2.extras import DictCursor
import uuid
from datetime import datetime
import os
from dotenv import load_dotenv
import logging

logger = logging.getLogger(__name__)

def validate_db_env_vars():
    required_vars = ['POSTGRES_DB', 'POSTGRES_USER', 'POSTGRES_PASSWORD', 'POSTGRES_HOST', 'POSTGRES_PORT']
    missing = [var for var in required_vars if not os.getenv(var)]
    if missing:
        raise EnvironmentError(f"Missing required database environment variables: {', '.join(missing)}")

def get_db_connection():
    """Create database connection"""
    load_dotenv()
    validate_db_env_vars()
    return psycopg2.connect(
        dbname=os.getenv('POSTGRES_DB'),
        user=os.getenv('POSTGRES_USER'),
        password=os.getenv('POSTGRES_PASSWORD'),
        host=os.getenv('POSTGRES_HOST'),
        port=os.getenv('POSTGRES_PORT')
    )

def setup_database(conn):
    """Create necessary tables if they don't exist"""
    logger.info("Setting up database tables...")
    
    create_table_query = """
    CREATE TABLE IF NOT EXISTS chat_interactions (
        id TEXT PRIMARY KEY,
        timestamp TIMESTAMP,
        question TEXT,
        answer TEXT,
        policy_type VARCHAR(50),
        intent VARCHAR(50),
        sentiment VARCHAR(50),
        complexity_level VARCHAR(50),
        response_status VARCHAR(50),
        response_type VARCHAR(50),
        user_feedback INTEGER,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """
    
    with conn.cursor() as cur:
        cur.execute(create_table_query)
        conn.commit()
    logger.info("Database setup completed")

def store_interaction(conn, **kwargs) -> uuid.UUID:
    interaction_id = uuid.uuid4()
    
    insert_query = """
    INSERT INTO chat_interactions (
        id, timestamp, question, answer,
        policy_type, intent, sentiment, complexity_level,
        response_status, response_type, user_feedback
    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """
    
    with conn.cursor() as cur:
        cur.execute(
            insert_query,
            (
                interaction_id, str(kwargs['id']), datetime.now(),
                kwargs['query'], kwargs['answer'], kwargs['policy_type'],
                kwargs['intent'], kwargs['sentiment'], kwargs['complexity_level'],
                kwargs['response_status'], kwargs['response_type'],
                kwargs.get('user_feedback')
            )
        )
        conn.commit()
    
    logger.info(f"Stored interaction with ID: {interaction_id}")
    return interaction_id

def update_user_feedback(conn, interaction_id: uuid.UUID, feedback: int) -> None:
    """Update user feedback for a specific interaction"""
    logger.info(f"Updating feedback for interaction: {interaction_id}")
    
    update_query = """
    UPDATE chat_interactions
    SET user_feedback = %s
    WHERE id = %s
    """
    
    with conn.cursor() as cur:
        cur.execute(update_query, (feedback, interaction_id))
        conn.commit()
    logger.info("Feedback updated successfully")