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
from prompt import build_prompt, is_farewell,is_greeting,default_template,response_templates


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
    """Setup Elasticsearch index with proper mappings"""
    index_name = "policy-questions"
    
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
    
    if not es_client.indices.exists(index=index_name):
        es_client.indices.create(index=index_name, body=index_settings)
        console.print("[green]Created Elasticsearch index 'policy-questions'[/green]")
        
        with open('data/documents-with-ids.json', 'r') as f:
            documents = json.load(f)
            
        for doc in documents:
            es_client.index(index=index_name, document=doc)
        
        console.print(f"[green]Indexed {len(documents)} documents[/green]")
    
    return index_name


def elastic_search(es_client, index_name, query, policy_type):
    """Search for relevant documents in Elasticsearch"""
    try:
        search_query = {
            "size": 5,
            "query": {
                "bool": {
                    "must": {
                        "multi_match": {
                            "query": query,
                            "fields": ["question^3", "text"],
                            "type": "best_fields"
                        }
                    }
                }
            }
        }
        
        # Only add policy filter if policy type is known and not "unknown"
        if policy_type and policy_type.lower() != "unknown":
            search_query["query"]["bool"]["filter"] = {
                "term": {
                    "policy": policy_type
                }
            }
            
        response = es_client.search(index=index_name, body=search_query)
        return [hit['_source'] for hit in response['hits']['hits']]
    except ConnectionError:
        logger.error("Unable to connect to Elasticsearch")
        return []
    except NotFoundError:
        logger.error("Elasticsearch index not found")
        return []
    except Exception as e:
        logger.error(f"Elasticsearch error: {str(e)}")
        return []

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
    """Detect policy type (e.g., auto, health, home) in customer query using OpenAI, if relevant."""
    prompt = """
    You are an AI assistant that detects the type of insurance policy referenced in a customer message.
    Identify the insurance policy type from the following options if present in the query:
    - auto
    - health
    - home

    
    If no specific policy is mentioned, return "unknown".
    
    Query: "{query}"
    """
    
    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt.format(query=query)}],
        temperature=0.3
    )
    
    policy_type = response.choices[0].message.content.strip().lower()

    # List of valid policy types
    valid_policies = ["auto", "health", "home","unknown"]
    
    # Default to 'unknown' if response is invalid or more than one word
    if policy_type not in valid_policies or len(policy_type.split()) > 1:
        policy_type = 'unknown'

    return policy_type
    
def get_llm_response(openai_client, prompt):
    """Get response from LLM with sentiment analysis."""
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error("Error getting LLM response: %s", e)
        return "I apologize, but I'm having trouble generating a response. Please try again or contact our customer service."