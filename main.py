from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any
import uvicorn
from openai import OpenAI
from elasticsearch import Elasticsearch
from enum import Enum
from langchain.memory import ConversationBufferMemory
# logger_setup.py
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)




from utils import (
    initialize_clients,
    setup_elasticsearch,
    elastic_search,
    get_intent,
    get_policy_type,
    get_llm_response
)

app = FastAPI(title="CIC Insurance Chatbot API")

# Initialize clients at startup
openai_client, es_client = initialize_clients()
index_name = setup_elasticsearch(es_client)
# Initialize conversation memory buffer
conversation_memory = ConversationBufferMemory()
class PolicyType(str, Enum):
    AUTO = "auto"
    HEALTH = "health"
    HOME = "home"
    UNKNOWN = "unknown"

class Intent(str, Enum):
    INQUIRY = "inquiry"
    CLAIM = "claim"
    POLICY = "policy"
    COVERAGE_CHECK = "coverage_check"
    COMPLAINT = "complaint"
    SUPPORT = "support"

class UserQuery(BaseModel):
    message: str
    policy_type: Optional[str] = None  # Changed default from "string" to None
    user_id: Optional[str] = None
    context: Optional[Dict[str, Any]] = None

class ChatResponse(BaseModel):
    response: str
    detected_intent: str
    detected_policy_type: str  # This will now always have a meaningful value
    requires_policy_type: bool
    additional_info: Optional[Dict[str, Any]] = None

# Intent-specific routes
@app.post("/customer-support")
async def handle_customer_support(query: UserQuery):
    """Handle customer support inquiries with enhanced personalization and escalation."""
    
    # Step 1: Add user message to memory
    conversation_memory.chat_memory.add_user_message(query.message)

    # Step 2: Detect intent and policy type
    intent = get_intent(openai_client, query.message)
    policy_type = get_policy_type(openai_client, query.message) if not query.policy_type else query.policy_type
    
    try:
        # Step 3: Retrieve relevant knowledge base information
        search_results = elastic_search(es_client, index_name, query.message, policy_type=policy_type)
        
        # Step 4: Generate response even if no exact knowledge base match
        context = conversation_memory.load_memory_variables({})['history']
        
        # Include knowledge base results if available
        knowledge_context = ""
        if search_results:
            knowledge_context = "\n".join([f"Q: {doc.get('question', 'N/A')}\nA: {doc.get('text', 'N/A')}" for doc in search_results])
        
        # Step 5: Generate response with fallback for no knowledge base results
        knowledge_base_section = f"Context from knowledge base:\n{knowledge_context}\n" if knowledge_context else ""
        
        prompt = (
            "You are CIC's General Insurance assistant. Act as a knowledgeable insurance advisor.\n"
            "CIC Insurance Group is a leading Cooperative Insurer in Africa, providing insurance and related financial services in Kenya, Uganda, South Sudan and Malawi."
            "If the Question is a greeting or an appreciation comment, respond accordingly in a polite manner but never step out of your context.\n\n"
            f"{knowledge_base_section}"
            f"Conversation history:\n{context}\n\n"
            f"Current query:\n{query.message}\n\n"
            "Follow these guidelines:\n"
            "1. If this is a general inquiry or greeting, respond appropriately without requiring specific policy information\n"
            "2. If the query can be answered using general insurance knowledge, provide a helpful response\n"
            "3. For product inquiries:\n"
            "    - Ask specific follow-up questions about the customer's needs\n"
            "    - Recommend products based on their requirements\n"
            "    - Highlight key benefits and differentiators\n"
            "4. If detecting purchase intent:\n"
            "    - Respond with 'purchase_intent_detected'\n"
            "    - Include the specific product(s) of interest\n"
            "5. Only escalate to an agent if the query requires specific policy holder information or complex claims handling\n\n"
            "Policy-specific information collection requirements:\n"
            "- Auto Insurance: Vehicle details, usage pattern, desired coverage level\n"
            "- Health Insurance: Age, medical history overview, desired coverage type\n"
            "- Home Insurance: Property type, location, desired coverage components"
        )

        response = get_llm_response(openai_client, prompt)

        # Step 6: Handle special cases
        if "purchase_intent_detected" in response:
            product_forms = {
                "auto": [
                    "- Full Name:",
                    "- Contact Number:",
                    "- Email Address:",
                    "- Vehicle Make and Model:",
                    "- Vehicle Year:",
                    "- Primary Vehicle Use (Personal/Commercial):",
                    "- Desired Coverage Level (Basic/Comprehensive):"
                ],
                "health": [
                    "- Full Name:",
                    "- Contact Number:",
                    "- Email Address:",
                    "- Age:",
                    "- Number of Family Members to Cover:",
                    "- Any Pre-existing Conditions:",
                    "- Preferred Coverage Type:"
                ],
                "home": [
                    "- Full Name:",
                    "- Contact Number:",
                    "- Email Address:",
                    "- Property Type:",
                    "- Property Address:",
                    "- Property Age:",
                    "- Desired Coverage Components:"
                ]
            }
            
            form_fields = product_forms.get(policy_type, product_forms["auto"])
            form_prompt = (
                f"I'll help you get started with your {policy_type} insurance! "
                f"Please provide the following information:\n\n"
                f"{chr(10).join(form_fields)}\n\n"
                "A CIC Insurance agent will contact you shortly to finalize your request."
            )
            
            conversation_memory.chat_memory.add_ai_message(form_prompt)
            return {
                "response": form_prompt,
                "intent": intent,
                "policy_type": policy_type,
                "status": "form_requested",
                "product_type": policy_type
            }

        # Step 7: Add assistant response to memory
        conversation_memory.chat_memory.add_ai_message(response)

        return {
            "response": response,
            "intent": intent,
            "policy_type": policy_type,
            "status": "completed"
        }

    except Exception as e:
        logger.error(f"Error in customer support handler: {str(e)}")
        # Provide a more helpful fallback response
        fallback_response = (
            "I understand your query, but I'm currently experiencing some technical difficulties. "
            "I can still help you with general insurance information or connect you with a specialist. "
            "Would you like to proceed with either option?"
        )
        return {
            "response": fallback_response,
            "intent": intent,
            "policy_type": policy_type,
            "status": "fallback"
        }
 

@app.post("/policy-info")
async def handle_policy_info(query: UserQuery):
    """Provide policy information to the user."""
    if not query.policy_type:
        raise HTTPException(status_code=400, detail="Policy type is required for policy information")
    
    relevant_docs = elastic_search(es_client, index_name, query.message, query.policy_type)
    context = "\n".join([doc["text"] for doc in relevant_docs])
    
    prompt = f"""
    Based on these policy details:
    {context}
    
    Please answer this policy question:
    {query.message}
    """
    response = get_llm_response(openai_client, prompt)
    
    return {
        "response": response,
        "intent": "policy",
        "policy_type": query.policy_type,
        "status": "completed"
    }

@app.post("/claims-processing")
async def handle_claims_status(query: UserQuery):
    """Check and return the claim status."""
    if not query.policy_type:
        raise HTTPException(status_code=400, detail="Policy type is required for claims status")
    
    prompt = f"""
    You are a claims specialist for {query.policy_type} insurance.
    Please provide information about this claims query:
    {query.message}
    """
    response = get_llm_response(openai_client, prompt)
    
    return {
        "response": response,
        "intent": "claim",
        "policy_type": query.policy_type,
        "status": "completed"
    }

@app.post("/coverage-check")
async def handle_coverage_check(query: UserQuery):
    """Check coverage details."""
    if not query.policy_type:
        raise HTTPException(status_code=400, detail="Policy type is required for coverage check")
    
    relevant_docs = elastic_search(es_client, index_name, query.message, query.policy_type)
    context = "\n".join([doc["text"] for doc in relevant_docs])
    
    prompt = f"""
    Based on these coverage details:
    {context}
    
    Please answer this coverage question:
    {query.message}
    """
    response = get_llm_response(openai_client, prompt)
    
    return {
        "response": response,
        "intent": "coverage_check",
        "policy_type": query.policy_type,
        "status": "completed"
    }

@app.post("/complaint-handling")
async def handle_complaint(query: UserQuery):
    """Handle customer complaints."""
    prompt = f"""
    You are a customer service specialist handling a complaint.
    
    Customer complaint:
    {query.message}
    
    Provide an empathetic response and clear next steps for resolution.
    """
    response = get_llm_response(openai_client, prompt)
    
    return {
        "response": response,
        "intent": "complaint",
        "policy_type": query.policy_type,
        "status": "completed"
    }

# Intent to endpoint mapping
INTENT_ENDPOINTS = {
    Intent.SUPPORT: "/customer-support",
    Intent.POLICY: "/policy-info",
    Intent.CLAIM: "/claims-processing",
    Intent.COVERAGE_CHECK: "/coverage-check",
    Intent.COMPLAINT: "/complaint-handling",
    Intent.INQUIRY: "/customer-support"
}

@app.post("/process-query")
async def process_query(query: UserQuery):
    """Main endpoint to process and route user queries."""
    try:
        # Detect intent
        detected_intent = get_intent(openai_client, query.message)    
        
        # Only use the detected policy type if none was provided or if "string" was provided
        if not query.policy_type or query.policy_type == "string":
            detected_policy_type = get_policy_type(openai_client, query.message)
        else:
            detected_policy_type = query.policy_type
        
        # Update the query object with the detected policy type
        query.policy_type = detected_policy_type
        
        # Get the appropriate endpoint
        endpoint = INTENT_ENDPOINTS.get(Intent(detected_intent))
        if not endpoint:
            raise HTTPException(status_code=400, detail="Invalid intent detected")
            
        # Route to appropriate handler
        if endpoint == "/customer-support":
            response = await handle_customer_support(query)
        elif endpoint == "/policy-info":
            response = await handle_policy_info(query)
        elif endpoint == "/claims-processing":
            response = await handle_claims_status(query)
        elif endpoint == "/coverage-check":
            response = await handle_coverage_check(query)
        elif endpoint == "/complaint-handling":
            response = await handle_complaint(query)
        else:
            raise HTTPException(status_code=400, detail="Invalid endpoint")
        
        return ChatResponse(
            response=response["response"],
            detected_intent=detected_intent,
            detected_policy_type=detected_policy_type,  # Use the detected policy type here
            requires_policy_type=False,
            additional_info={
                "status": response["status"],
                "endpoint_used": endpoint
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)