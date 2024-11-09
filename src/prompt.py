# prompt.py

from typing import List, Dict
import re


# Define response templates for each insurance type
response_templates = {
    "Health Insurance": """
    You are CIC's Health Insurance assistant. CIC provides a range of health insurance
    policies, including coverage for medical treatments, hospital visits, and prescription
    drugs. Answer the customer's health insurance questions using the context below, and 
    consider these guidelines:
    
    1. Offer clarity on coverage details, deductibles, and limits related to health.
    2. If relevant, advise on hospital networks or emergency services.
    3. If specific coverage isn't available, politely guide the customer to contact support.
    
    """,
    
    "Auto Insurance": """
    You are CIC's Auto Insurance assistant. CIC offers auto insurance policies, covering
    vehicle damage, accidents, theft, and more. Respond to the customer’s auto insurance
    inquiries using the provided context, following these guidelines:
    
    1. Focus on vehicle coverage details, accident protocols, and repair processes.
    2. If relevant, provide information on claims for different types of damage.
    3. If exact coverage is unclear, suggest the customer contacts our support team.
    
    """,
    
    "Home Insurance": """
    You are CIC's Home Insurance assistant. CIC provides home insurance policies that cover
    property damage, theft, and personal liability. Address the customer's home insurance
    questions using the context below, keeping in mind:
    
    1. Include information on property coverage limits, theft, and damage coverage.
    2. Address concerns regarding structural damage, loss, or liability coverage.
    3. If unsure, recommend the customer reaches out to our support for detailed help.
    
    """
}

# Default template if no specific policy type is detected
default_template = """
You are CIC's General Insurance assistant. CIC covers health, auto, and home insurance.
Respond professionally to the customer's query. Use context to answer their question, and
if unclear, kindly suggest contacting support for specific details.
"""

# Helper functions
def is_greeting(query: str) -> bool:
    """Check if the input is a greeting or conversation starter."""
    greetings = [
        r'\b(hi|hello|hey|good\s?(morning|afternoon|evening)|greetings)\b',
        r'\bhow\s?are\s?you\b',
        r'\bhi\s?there\b',
        r'\bhey\s?there\b',
        r'\bgood\s?day\b',
        r'\bnice\s?to\s?meet\s?you\b',
        r'\bcan\s?you\s?help\b',
        r'\bassistance\s?needed\b'
    ]
    return any(re.search(pattern, query.lower()) for pattern in greetings)

def is_farewell(query: str) -> bool:
    """Check if the input is a farewell message."""
    farewells = [
        r'\b(goodbye|bye|see\s?you|farewell|thanks|thank\s?you|appreciate\s?it)\b',
        r'\bhave\s?a\s?(good|nice|great)\s?(day|evening|night|time)\b'
    ]
    return any(re.search(pattern, query.lower()) for pattern in farewells)

# Main function to build prompt
def build_prompt(query: str, search_results: List[Dict], policy_type: str, intent: str, sentiment: str) -> str:
    """Build enhanced prompt for the LLM with improved context awareness."""
    # Get the appropriate response template
    template = response_templates.get(policy_type, default_template)

    # Greeting and Farewell prompts
    if is_greeting(query):
        return f"{template}\n\nCUSTOMER GREETING: {query}\n\nPlease provide your response:"
    elif is_farewell(query):
        return f"{template}\n\nCUSTOMER FAREWELL: {query}\n\nPlease provide your response:"

    # Build context from search results
    context = ""
    for doc in search_results:
        context += f"\nQ: {doc.get('question', 'N/A')}\nA: {doc.get('text', 'N/A')}\n"

    # Main prompt for general queries
    return f"""
    {template}
    
    POLICY TYPE: {policy_type}
    CUSTOMER INTENT: {intent}
    CUSTOMER SENTIMENT: {sentiment}
    
    CUSTOMER QUERY: {query}
    
    RELEVANT CONTEXT:
    {context}
    
    Please provide your response:
    """


