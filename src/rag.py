from elasticsearch import Elasticsearch
from elasticsearch.exceptions import ConnectionError, NotFoundError
import logging
import os
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

def validate_env_vars():
    if not os.getenv('ELASTICSEARCH_HOST'):
        raise EnvironmentError("Missing ELASTICSEARCH_HOST environment variable")

def setup_index(es_client, index_name: str) -> None:
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
                "question": {
                    "type": "text",
                    "analyzer": "custom_analyzer"
                },
                "text": {
                    "type": "text",
                    "analyzer": "custom_analyzer"
                },
                "policy_type": {
                    "type": "keyword"
                },
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
        load_dotenv()
        validate_env_vars()
        es_client = Elasticsearch(os.getenv('ELASTICSEARCH_HOST'))

        if es_client.indices.exists(index=index_name):
            es_client.indices.delete(index=index_name)
            logger.info(f"Deleted existing index: {index_name}")

        es_client.indices.create(
            index=index_name,
            body=index_settings
        )
        logger.info(f"Created new index: {index_name}")
    except Exception as e:
        logger.error(f"Error setting up index: {e}")
        raise

def search_documents(es_client, query: str, policy_type: str, index_name: str) -> list:
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
        results = [hit['_source'] for hit in response['hits']['hits']]
        logger.info(f"Found {len(results)} relevant documents")
        return results
    except Exception as e:
        logger.error(f"Search error: {e}")
        return []