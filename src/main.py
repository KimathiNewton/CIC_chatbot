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
from rag import setup_index, search_documents
from db import setup_database, store_interaction, update_user_feedback,get_db_connection
from ingest import get_es_client,setup_es_index
import uuid

# Initialize logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
console = Console()

# Initialize Rich console for better output formatting
console = Console()


def get_time_of_day() -> str:
    """Return appropriate greeting based on time of day."""
    hour = datetime.now().hour
    if 5 <= hour < 12:
        return "morning"
    elif 12 <= hour < 17:
        return "afternoon"
    else:
        return "evening"
    



def get_greeting_response(query: str) -> str:
    """Generate contextual greeting response based on query type and time of day."""
    time_of_day = get_time_of_day()
    
    if re.search(r'\bhow\s?are\s?you\b', query.lower()):
        return (
            f"Good {time_of_day}! I'm doing well, thank you for asking. "
            "As CIC's insurance assistant, I'm here to help you with any insurance-related "
            "questions about our health, auto, or home insurance policies. How may I assist you today?"
        )
    elif re.search(r'\bcan\s?you\s?help\b', query.lower()):
        return (
            f"Good {time_of_day}! Absolutely, I'd be happy to help! "
            "I'm CIC's insurance assistant, specializing in health, auto, and home insurance policies. "
            "Please let me know what you need assistance with."
        )
    else:
        return (
            f"Good {time_of_day}! Welcome to CIC Insurance. I'm your dedicated insurance assistant, "
            "here to help you with any questions about our health, auto, or home insurance policies. "
            "How can I assist you today?"
        )

def get_farewell_response() -> str:
    """Generate appropriate farewell response."""
    return (
        "Thank you for chatting with CIC Insurance. If you have any more questions, "
        "don't hesitate to ask. Have a great rest of your day!"
    )

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





def detect_policy_type(query: str) -> str:
    """Detect insurance policy type from query"""
    policy_keywords = {
        "Health Insurance": [
            "health", "medical", "doctor", "hospital", "prescription",
            "clinic", "treatment", "diagnosis", "healthcare"
        ],
        "Auto Insurance": [
            "car", "vehicle", "auto", "accident", "collision",
            "motor", "driving", "driver", "traffic"
        ],
        "Home Insurance": [
            "home", "house", "property", "damage", "theft",
            "building", "residential", "apartment", "condo"
        ]
    }
    
    query_lower = query.lower()
    for policy_type, keywords in policy_keywords.items():
        if any(keyword in query_lower for keyword in keywords):
            return policy_type
    return "General Insurance" 



def analyze_complexity(query: str, policy_type: str) -> Tuple[bool, str]:
    """Analyze query complexity based on various factors."""
    complex_indicators = {
        "multiple_policies": [
            "policies", "both", "all", "combine", "multiple",
            "different", "various", "several"
        ],
        "cross_departmental": [
            "department", "team", "specialist", "expert", "supervisor",
            "manager", "authority", "administration"
        ],
        "specialized_input": {
            "Health Insurance": [
                "pre-existing", "condition", "medical history", "assessment",
                "underwriting", "specialist approval", "chronic", "disability"
            ],
            "Auto Insurance": [
                "inspection", "evaluation", "custom", "modification",
                "commercial use", "fleet", "multi-car", "classic car"
            ],
            "Home Insurance": [
                "valuation", "assessment", "renovation", "security system",
                "high-value", "commercial use", "listed building", "contents"
            ]
        }
    }
    
    query_lower = query.lower()
    
    if any(word in query_lower for word in complex_indicators["multiple_policies"]):
        return True, "Query involves multiple policies"
        
    if any(word in query_lower for word in complex_indicators["cross_departmental"]):
        return True, "Query requires cross-departmental support"
        
    if policy_type in complex_indicators["specialized_input"]:
        if any(term in query_lower for term in complex_indicators["specialized_input"][policy_type]):
            return True, f"Query requires specialized {policy_type} assessment"
            
    return False, ""

def get_sentiment(openai_client, query: str) -> str:
    """Analyze sentiment of customer query using OpenAI."""
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "Analyze the sentiment and return ONLY ONE WORD from these options: positive, negative, neutral, frustrated, confused, urgent"},
                {"role": "user", "content": f"Analyze this message: '{query}'"}
            ],
            temperature=0.3
        )
        # Extract just the sentiment word and clean it
        sentiment = response.choices[0].message.content.strip().lower()
        # If response is longer than one word, default to 'neutral'
        if len(sentiment.split()) > 1:
            sentiment = 'neutral'
        return sentiment
    except Exception as e:
        logger.error(f"Error getting sentiment analysis: {e}")
        return "neutral"

# Modify the get_llm_response function to include sentiment analysis
def get_llm_response(openai_client, prompt):
    """Get response from LLM with sentiment analysis."""
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error("Error getting LLM response: %s", e)
        return "I apologize, but I'm having trouble generating a response. Please try again or contact our customer service."
def get_intent(openai_client, query: str, policy_type: str) -> str:
    """Classify intent of customer query using OpenAI."""
    try:
        prompt = f"""
        You are an AI assistant that classifies the intent of customer messages.
        Classify the customer's intent for this {policy_type} insurance query into ONE of these categories:
        - inquiry
        - claim
        - renewal
        - coverage_check
        - complaint
        - update
        - emergency
        
        Return ONLY the category word, nothing else.
        
        Query: "{query}"
        """
        
        response = openai_client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        intent = response.choices[0].message.content.strip().lower()
        # If response is more than one word, default to 'inquiry'
        if len(intent.split()) > 1:
            intent = 'inquiry'
        return intent
    except Exception as e:
        logger.error(f"Error getting intent classification: {e}")
        return "inquiry"




def detect_severity(query: str) -> Tuple[bool, str]:
    """Detect if the query involves severe incidents or urgent situations."""
    severity_indicators = {
        "health": [
            "emergency", "accident", "injured", "hospital", "ambulance",
            "critical", "severe pain", "life-threatening", "urgent care",
            "serious condition", "intensive care"
        ],
        "property": [
            "fire", "flood", "break-in", "theft", "major damage",
            "structural damage", "disaster", "emergency repairs",
            "security breach", "vandalism"
        ],
        "vehicle": [
            "crash", "collision", "totaled", "major accident",
            "hit and run", "stolen", "wreck", "rollover",
            "multi-vehicle", "severe damage"
        ]
    }
    
    query_lower = query.lower()
    
    for category, indicators in severity_indicators.items():
        if any(indicator in query_lower for indicator in indicators):
            return True, f"Severe {category}-related incident detected"
            
    return False, ""

def analyze_customer_emotion(query: str) -> Tuple[bool, str]:
    """Analyze customer emotion in the query."""
    emotion_indicators = {
        "urgency": [
            "urgent", "emergency", "asap", "immediately", "right now",
            "cannot wait", "time sensitive", "crucial", "pressing"
        ],
        "distress": [
            "worried", "concerned", "anxious", "scared", "stressed",
            "frustrated", "desperate", "help", "fearful", "uncertain"
        ],
        "dissatisfaction": [
            "unhappy", "disappointed", "unsatisfied", "complaint",
            "upset", "angry", "horrible", "terrible", "unacceptable",
            "poor service", "not working"
        ]
    }
    
    query_lower = query.lower()
    
    for emotion, indicators in emotion_indicators.items():
        if any(indicator in query_lower for indicator in indicators):
            return True, f"Detected customer {emotion}"
            
    return False, ""

def needs_escalation(query: str, answer: str, policy_type: str, sentiment: str = "neutral") -> Tuple[bool, str]:
    """Enhanced escalation check based on multiple criteria."""
    # Skip escalation for greetings and farewells
    if is_greeting(query) or is_farewell(query):
        return False, ""
        
    # Check severity
    is_severe, severity_reason = detect_severity(query)
    if is_severe:
        return True, severity_reason
        
    # Check complexity
    is_complex, complexity_reason = analyze_complexity(query, policy_type)
    if is_complex:
        return True, complexity_reason
        
    # Check sentiment for escalation
    if sentiment in ["frustrated", "urgent", "negative"]:
        return True, f"Customer showing {sentiment} sentiment"
        
    # Check for system uncertainty
    uncertainty_phrases = [
        'contact customer service',
        'cannot provide',
        'don\'t have information',
        'unable to assist',
        'not authorized',
        'need more information',
        'unclear',
        'unsure',
        'cannot determine'
    ]
    if any(phrase in answer.lower() for phrase in uncertainty_phrases):
        return True, "System unable to provide confident response"
        
    return False, ""



def chat_loop():
    """Main chat loop with proper resource management"""
    try:
        # Initialize connections and resources
        id = str(uuid.uuid4())
        db_conn = get_db_connection()
        es_client = get_es_client()
        openai_client = OpenAI()

        # Setup database and elasticsearch
        setup_database(db_conn)
        setup_es_index(es_client, "policy-questions")

        console.print(Panel.fit(
            "[bold blue]Welcome to CIC Insurance Virtual Assistant[/bold blue]\n"
            "I'm here to help you with your insurance questions.\n"
            "Type 'exit' to end our conversation.",
            title="👋 Welcome to CIC Insurance",
            border_style="blue"
        ))

        while True:
            query = console.input("\n[bold green]You:[/bold green] ").strip()

            if not query:
                continue

            if query.lower() in ['exit', 'quit', 'bye', 'goodbye']:
                farewell = get_farewell_response()
                console.print("\n[bold blue]Assistant:[/bold blue]", style="bold")
                console.print(farewell)
                store_interaction(
                    db_conn,
                    id=id,
                    query=query,
                    answer=farewell,
                    policy_type="General",
                    intent="farewell",
                    sentiment="neutral",
                    complexity_level="simple",
                    response_status="normal",
                    response_type="farewell"
                )
                break

            # Process query
            policy_type = detect_policy_type(query)

            # Handle greetings
            if is_greeting(query):
                answer = get_greeting_response(query)
                store_interaction(
                    db_conn,
                    id=id,
                    query=query,
                    answer=answer,
                    policy_type=policy_type,
                    intent="greeting",
                    sentiment="neutral",
                    complexity_level="simple",
                    response_status="normal",
                    response_type="greeting"
                )
                console.print("\n[bold blue]Assistant:[/bold blue]", style="bold")
                console.print(answer)
                continue

            # Regular query processing
            intent = get_intent(openai_client, query, policy_type)
            sentiment = get_sentiment(openai_client, query)

            # Get relevant documents and generate response
            search_results = search_documents(es_client, query, policy_type, "policy-questions")
            prompt = build_prompt(query, search_results, policy_type, intent, sentiment)
            answer = get_llm_response(openai_client, prompt)

            # Check for escalation
            needs_escalate, reason = needs_escalation(query, answer, policy_type, sentiment)

            # Store interaction
            store_interaction(
                db_conn,
                id=id,
                query=query,
                answer=answer,
                policy_type=policy_type,
                intent=intent,
                sentiment=sentiment,
                complexity_level="complex" if needs_escalate else "simple",
                response_status="escalated" if needs_escalate else "normal",
                response_type="query"
            )

            # Display response
            console.print("\n[bold blue]Assistant:[/bold blue]", style="bold")
            console.print(Markdown(answer))

            # Handle escalation if needed
            if needs_escalate:
                escalation_message = (
                    f"\n[bold red]⚠️ Important:[/bold red] This query requires special attention ({reason}). "
                    "\n• A customer service representative will contact you shortly."
                    "\n• For urgent matters, please contact our 24/7 support line:"
                    "\n  [bold]📞 0800-CIC-HELP[/bold]"
                    "\n  [bold]📧 support@cic.com[/bold]"
                )
                console.print(Panel(escalation_message, style="red", title="Escalation Notice"))

                # Log escalation
                logger.info(f"Query escalated: {reason} - Query: {query}")

    except KeyboardInterrupt:
        console.print("\n[yellow]Chat session ended by user.[/yellow]")
    except Exception as e:
        logger.error(f"Error in chat loop: {e}")
        console.print(f"\n[red]An error occurred: {str(e)}[/red]")
    finally:
        # Clean up resources
        if 'db_conn' in locals():
            db_conn.close()
        if 'es_client' in locals():
            es_client.close()

if __name__ == "__main__":
    chat_loop()