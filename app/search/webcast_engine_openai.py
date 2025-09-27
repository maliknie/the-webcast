# File: webcast_engine.py

import os
from openai import OpenAI

# --- Configuration ---
# It's best practice to load secrets like API keys from environment variables.
# This keeps them out of your source code.
API_KEY = os.getenv("SWISS_AI_PLATFORM_API_KEY")
print(f"{API_KEY=}")
API_URL = "https://api.swiss-ai-platform.ch/v1/chat/completions" # As per the guide


STRICT_SYSTEM = """You are WebCast, a grounded assistant. When CONTEXT is provided:
- Treat it as the ONLY source of truth.
- Do NOT use prior knowledge if it conflicts with CONTEXT.
- Do NOT write hedging like "as of my last update", "I cannot browse", or "check the latest".
- Provide clear, factual answers based on the context.
- If a needed fact is missing, say: "Not in sources; cannot confirm."
Return a clear, concise answer. No extra text."""

USER_WITH_CONTEXT = """QUESTION:{query}

CONTEXT (sources you MUST rely on):
{context}

Provide a clear, factual answer based on the context above."""

USER_NO_CONTEXT = """QUESTION:
{query}"""


HEDGING = [
    "as of my last update",
    "as of the latest information",
    "I don't have access to the internet",
    "I cannot browse",
    "check the latest updates",
    "please check the latest",
]


def validate_response(response: str) -> tuple[bool, list[str]]:
    """
    Validates a response against hedging patterns and other quality checks.
    
    Args:
        response: The LLM response to validate
        
    Returns:
        A tuple of (is_valid, list_of_issues)
    """
    issues = []
    response_lower = response.lower()
    
    # Check for hedging patterns
    for hedge in HEDGING:
        if hedge.lower() in response_lower:
            issues.append(f"Contains hedging: '{hedge}'")
    
    return len(issues) == 0, issues



def get_llm_summary(user_query: str, context: str | None) -> str:
    """
    Calls the Apertus LLM on the Swiss AI Platform with an augmented prompt.

    Args:
        user_query: The original question from the user.
        context: The real-time information fetched from the web. 
                 If None, the query is passed directly without context.

    Returns:
        A string containing the LLM's generated answer.
        Returns an error message if the API call fails.
    """
    if not API_KEY:
        return "ERROR: SWISS_AI_PLATFORM_API_KEY environment variable not set."

    # 1. Construct the final prompt based on whether context is available
    if context:
        # Use the integrated prompt template for context-based queries
        final_prompt = USER_WITH_CONTEXT.format(query=user_query, context=context)
        system_prompt = STRICT_SYSTEM
    else:
        # Use the integrated prompt template for general queries
        final_prompt = USER_NO_CONTEXT.format(query=user_query)
        system_prompt = (
            "You are 'The WebCast', a helpful AI assistant that provides clear, factual summaries. "
            "Your goal is to answer the user's question accurately and concisely."
        )

    client = OpenAI(
        api_key=os.getenv("SWISS_AI_PLATFORM_API_KEY"),
        base_url="https://api.swisscom.com/layer/swiss-ai-weeks/apertus-70b/v1",
    )

    stream = client.chat.completions.create(
        model="swiss-ai/Apertus-70B",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": final_prompt}
        ],
        stream=True,
        max_tokens=300,
        temperature=0.1,
    )

    collected_chunks: list[str] = []
    for chunk in stream:
        token = chunk.choices[0].delta.content or ""
        if token:
            collected_chunks.append(token)

    response = "".join(collected_chunks).strip()
    
    # Validate the response for quality issues
    is_valid, issues = validate_response(response)
    if not is_valid and context:  # Only warn for context-based queries
        print(f"Warning: Response quality issues detected: {', '.join(issues)}")
    
    return response



# # --- Example Usage ---
# # This block allows you to test this file directly before integrating it.
# if __name__ == "__main__":
#     print("=== WebCast Engine with Integrated Prompts ===\n")
    
#     print("--- Test 1: Query WITH real-time context (RAG mode) ---")
#     mock_user_query = "What is the latest iphone released?"
#     mock_context = """iphone 17 """
    
#     summary = get_llm_summary(user_query=mock_user_query, context=mock_context)
#     print(f"User Query: {mock_user_query}")
#     print(f"LLM Summary:\n{summary}\n")

#     print("--- Test 2: Query WITHOUT real-time context (General mode) ---")
#     general_query = "What is the latest iphone released?"
    
#     summary_general = get_llm_summary(user_query=general_query, context=None)
#     print(f"User Query: {general_query}")
#     print(f"LLM Summary:\n{summary_general}\n")

#     print("--- Test 3: API Key validation ---")
#     # Temporarily unset the API_KEY to test the error handling
#     original_key = API_KEY
#     API_KEY = None
#     error_summary = get_llm_summary(user_query="test", context="test")
#     print(f"Error handling result: {error_summary}")
#     API_KEY = original_key  # Reset key
    
#     print("\n--- Test 4: Response validation demonstration ---")
#     # Test with a query that might trigger hedging
#     test_query = "What's the latest news about iphone developments?"
#     test_context = "Recent AI developments include new language models and improved reasoning capabilities."
    
#     validated_summary = get_llm_summary(user_query=test_query, context=test_context)
#     print(f"User Query: {test_query}")
#     print(f"LLM Summary:\n{validated_summary}")