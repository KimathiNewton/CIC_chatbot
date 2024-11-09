import os
import json
import logging
from typing import List, Dict
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def validate_es_env_vars():
    if not os.getenv('ELASTICSEARCH_HOST'):
        raise EnvironmentError("Missing ELASTICSEARCH_HOST environment variable")

def get_es_client():
    """Initialize and return Elasticsearch client"""
    load_dotenv()
    validate_es_env_vars()
    return Elasticsearch(os.getenv('ELASTICSEARCH_HOST'))

def load_data(data_path: str = "data/documents-with-ids.json") -> List[Dict]:
    """Load and preprocess data from JSON file"""
    try:
        with open(data_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            if isinstance(data, dict):
                data = list(data.values())[0] if len(data) == 1 else [data]
            logger.info(f"Loaded {len(data)} documents from JSON")
            return data
    except Exception as e:
        logger.error(f"Error loading JSON data: {e}")
        return []

def prepare_documents(documents: List[Dict], index_name: str) -> List[Dict]:
    """Prepare documents for Elasticsearch indexing"""
    prepared_docs = []
    for doc in documents:
        prepared_doc = {
            "_index": index_name,
            "_source": {
                "question": doc.get("question", ""),
                "text": doc.get("answer", ""),
                "policy_type": doc.get("policy_type", "general"),
                "metadata": {
                    "id": doc.get("id", ""),
                    "category": doc.get("category", "")
                }
            }
        }
        prepared_docs.append(prepared_doc)
    return prepared_docs

def setup_es_index(es_client, index_name: str) -> None:
    """Setup Elasticsearch index with proper mappings"""
    index_settings = {
        "settings": {
            "number_of_shards": 1,
            "number_of_replicas": 0,
            "analysis": {
                "analyzer": {
                    "custom_analyzer": {
                        "type": "custom",
                        "tokenizer": "standard",
                        "filter": ["lowercase", "stop", "snowball", "word_delimiter_graph"]
                    }
                }
            }
        },
        "mappings": {
            "properties": {
                "question": {"type": "text", "analyzer": "custom_analyzer"},
                "text": {"type": "text", "analyzer": "custom_analyzer"},
                "policy_type": {"type": "keyword"},
                "metadata": {
                    "properties": {
                        "id": {"type": "keyword"},
                        "category": {"type": "keyword"}
                    }
                }
            }
        }
    }

    try:
        if es_client.indices.exists(index=index_name):
            es_client.indices.delete(index=index_name)
        es_client.indices.create(index=index_name, body=index_settings)
        logger.info(f"Created index: {index_name}")
    except Exception as e:
        logger.error(f"Error setting up index: {e}")
        raise

def search_documents(es_client, query: str, policy_type: str, index_name: str) -> List[Dict]:
    """Search for relevant documents"""
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
                    },
                    "filter": {
                        "term": {
                            "policy_type": policy_type
                        }
                    }
                }
            }
        }
        
        response = es_client.search(index=index_name, body=search_query)
        return [hit['_source'] for hit in response['hits']['hits']]
    except Exception as e:
        logger.error(f"Search error: {e}")
        return []