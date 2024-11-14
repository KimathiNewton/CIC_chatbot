import os
from openai import OpenAI
from elasticsearch import Elasticsearch
from dotenv import load_dotenv
import json
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
import logging
from elasticsearch.exceptions import ConnectionError, NotFoundError
import re
from typing import Tuple, List, Dict
from datetime import datetime
from tqdm.auto import tqdm


# Initialize Rich console for better output formatting
console = Console()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def initialize_clients():
    """Initialize OpenAI and Elasticsearch clients."""
    try:
        load_dotenv()
        openai_client = OpenAI()
        es_client = Elasticsearch('http://localhost:9200')
        logger.info("Clients initialized successfully")
        return openai_client, es_client
    except Exception as e:
        logger.error("Failed to initialize clients: %s", e)
        raise

def setup_elasticsearch(es_client):
    """
    Sets up an Elasticsearch index with given settings and indexes documents from a JSON file.

    Args:
        es_client (Elasticsearch): The Elasticsearch client instance.
        json_file_path (str): Path to the JSON file containing documents.
        index_name (str): Name of the Elasticsearch index to create and index documents into.

    Returns:
        str: The name of the Elasticsearch index.
    """
    # Define index settings and mappings
    index_settings = {
        "settings": {
            "number_of_shards": 1,
            "number_of_replicas": 0
        },
        "mappings": {
            "properties": {
                "text": {"type": "text"},
                "question": {"type": "text"},
                "policy": {"type": "keyword"}
            }
        }
    }
    index_name="policy-questions"
    # Delete the index if it exists, then create it with the specified settings
    es_client.indices.delete(index=index_name, ignore_unavailable=True)
    es_client.indices.create(index=index_name, body=index_settings)
    
    # Load document data from JSON file
    with open('data/documents-with-ids.json', 'rt') as f:
        documents = json.load(f)

    # Index each document
    for doc in documents:
        es_client.index(index=index_name, document=doc)

    return index_name


def elastic_search(es_client, index_name, query, policy_type):
    """
    Performs a search query on an Elasticsearch index, with a policy filter if a policy type is specified.
    
    Args:
        es_client (Elasticsearch): The Elasticsearch client instance.
        index_name (str): The name of the Elasticsearch index to search.
        query (str): The search query string.
        policy_type (str): The policy term to filter results by, if detected.
    
    Returns:
        list: A list of documents that match the search criteria.
    """
    # Define the base search query
    search_query = {
        "size": 5,
        "query": {
            "bool": {
                "must": {
                    "multi_match": {
                        "query": query,
                        "fields": ["question^3", "text", "section"],
                        "type": "best_fields"
                    }
                }
            }
        }
    }
    
    # Add policy filter only if a specific policy type was detected
    if policy_type != "Unknown":
        search_query["query"]["bool"]["filter"] = {
            "term": {
                "policy": policy_type
            }
        }

    # Execute the search query
    response = es_client.search(index=index_name, body=search_query)
    
    # Extract the results
    result_docs = [hit['_source'] for hit in response['hits']['hits']]
    
    return result_docs




def get_intent(openai_client, query: str) -> str:
    """Classify intent of customer query using OpenAI."""
    prompt = """
    You are an AI assistant that classifies the intent of customer messages.
    Classify the customer's intent into ONE of these categories:
    - inquiry
    - claim
    - Policy 
    - coverage_check
    - complaint
    - Support

    
    Return ONLY the category word, nothing else.
    
    Query: "{query}"
    """

    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt.format(query=query)}],
        temperature=0.3
    )
    
    intent = response.choices[0].message.content.strip().lower()

    # Valid intents to check against
    valid_intents = ["inquiry", "claim", "Policy", "coverage_check", "complaint", "Support"]
    
    # Default to 'inquiry' if response is invalid or more than one word
    if intent not in valid_intents or len(intent.split()) > 1:
        intent = 'inquiry'

    return intent



def get_policy_type(openai_client, query: str) -> str:
    """Detect policy type (e.g., Auto Insurance, Health Insurance, Home Insurance) in customer query using OpenAI, if relevant."""
    prompt = """
    You are an AI assistant that detects the type of insurance policy referenced in a customer message.
    Identify the insurance policy type from the following options if present in the query and return exactly as written here:
    - Auto Insurance
    - Health Insurance
    - Homeowners Insurance
    - Unknown

    
    
    Query: "{query}"
    """
    
    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt.format(query=query)}],
        temperature=0.3
    )
    
    policy_type = response.choices[0].message.content.strip()
    return policy_type
    
def get_llm_response(openai_client, prompt, original_message: str):
    """Get response from LLM with intelligent status detection."""
    try:
        # First get the main response
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        response_text = response.choices[0].message.content

        # Now analyze the conversation status
        status_prompt = """
        As an AI assistant for CIC Insurance, analyze the conversation and determine its current status.
        
        Original User Message: {original_message}
        AI Response: {response}
        
        Determine the conversation status based on these criteria:
        
        1. ESCALATED: If the situation requires human agent intervention, such as:
           - Complex claims processing
           - Specific policy holder information needed
           - Customer explicitly requests human agent
           - Complaints that need human handling
           - Technical issues beyond AI capability
           - Auto Insurance purchase completion, once the User confirms the Purchase.
           - For Home Insurance, it escalated only for complex inquiries, that are serious (e.g., concerns about flood, fire coverage, high-value property)
           - Situations requiring policy verification
           - Cases needing document processing
        
        2. PENDING: If waiting for specific information from the user, such as:
           - Required details for policy quotation
           - Documentation or information needed to proceed
           - Verification details needed
           - Clarification needed about the query
        
        3. COMPLETED: If the conversation has reached a natural conclusion, and the user has confirmed the Plan they want to purchase:
           - Query fully answered with no follow-up needed
           - Process successfully completed
           - Customer indicates satisfaction with response
           - Health insurance purchase process completed online
           - Home Insurance Purchase that involves basic or general coverage, those that seem simple (e.g., standard property protection, coverage for personal belongings)
        
        4. ONGOING: If the conversation is active but:
           - Still in information gathering phase
           - Multiple steps remaining in the process
           - Further discussion needed
           - Customer might have follow-up questions
        
        Return only one of these exact words: ESCALATED, PENDING, COMPLETED, or ONGOING
        
        Additional Context: 
        - Health Insurance queries can be completed online without escalation
        - Auto Home Insurance always require escalation for purchase completion
        - Home Insurance might require escalation for purchase completion and in other cases the purchase is completed online without escalation
        - Claims processing typically requires escalation unless it's a simple status check
        """
        
        status_response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[{
                "role": "user", 
                "content": status_prompt.format(
                    original_message=original_message,
                    response=response_text
                )
            }],
            temperature=0.3  # Lower temperature for more consistent status detection
        )
        
        status = status_response.choices[0].message.content.strip().upper()
        
        # Validate status
        valid_statuses = {"ESCALATED", "PENDING", "COMPLETED", "ONGOING"}
        if status not in valid_statuses:
            status = "ONGOING"  # Default to ongoing if invalid status
            
        return response_text, status.lower()
        
    except Exception as e:
        logger.error("Error getting LLM response: %s", e)
        return "I apologize, but I'm having trouble generating a response. Please try again or contact our customer service.", "error"
