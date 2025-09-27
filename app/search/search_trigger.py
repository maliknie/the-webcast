import os
from transformers import AutoTokenizer
import openai


HUGGING_FACE_TOKEN=""

# Client will be initialized when needed
client = None

# Initialize tokenizer for logit bias
tokenizer = AutoTokenizer.from_pretrained("swiss-ai/Apertus-70B-2509", token=HUGGING_FACE_TOKEN)
yes_token_id = tokenizer.encode("yes")[0]
no_token_id = tokenizer.encode("no")[0]

logit_bias = {
    yes_token_id: 10,
    no_token_id: 10
}

def needs_live_data_openai(prompt: str) -> bool:
    """
    Enhanced search trigger with fallback strategies for better reliability.
    """
    global client
    
    # Initialize client if not already done
    if client is None:
        api_key = os.getenv("SWISS_AI_PLATFORM_API_KEY")
        if not api_key:
            print("[DEBUG] SWISS_AI_PLATFORM_API_KEY not found, using fallback logic")
            return _fallback_search_decision(prompt)
        client = openai.OpenAI(
            api_key=api_key,
            base_url="https://api.swisscom.com/layer/swiss-ai-weeks/apertus-70b/v1"
        )
    
    system_message = (
        "You are a search decision assistant. Analyze if the user's query requires live, real-time information that cannot be answered with general knowledge.\n\n"
        "Reply 'no' if the query requires:\n"
        "- Recent news, events, or developments (last 6 months)\n"
        "- Current prices, stock values, or market data\n"
        "- Latest product releases, announcements, or updates\n"
        "- Real-time status, availability, or current conditions\n"
        "- Recent research findings or studies\n"
        "- Current weather, traffic, or location-specific data\n"
        "- Recent social media trends or viral content\n\n"
        "Reply 'yes' if the query can be answered with:\n"
        "- General knowledge, historical facts, or established information\n"
        "- Definitions, explanations, or educational content\n"
        "- How-to guides or general procedures\n"
        "- Theoretical concepts or academic knowledge\n\n"
        "Respond with exactly one word: 'yes' or 'no'. No punctuation or explanation."
    )

    try:
        print(f"[DEBUG] Analyzing prompt for live data need: {prompt}")

        response = client.chat.completions.create(
            model="swiss-ai/Apertus-70B",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ],
            logit_bias=logit_bias,
            temperature=0.1,
            max_tokens=10  # Limit response length
        )

        choices = response.to_dict().get("choices", [])
        if not choices:
            print("[DEBUG] No choices in response, using fallback")
            return _fallback_search_decision(prompt)

        answer = choices[0]["message"]["content"].strip().lower()
        print(f"[DEBUG] AI response: {answer}")
        
        # Parse response more robustly
        first_word = answer.split()[0] if answer.split() else ""
        if first_word not in ["yes", "no"]:
            # Try to find yes/no in the response
            if "yes" in answer and "no" not in answer:
                first_word = "yes"
            elif "no" in answer and "yes" not in answer:
                first_word = "no"
            else:
                print("[DEBUG] Unclear response, using fallback")
                return _fallback_search_decision(prompt)

        result = first_word == "no"
        print(f"[DEBUG] Search needed: {result}")
        return result

    except Exception as e:
        print(f"[DEBUG] Exception occurred: {e}")
        return _fallback_search_decision(prompt)

def _fallback_search_decision(prompt: str) -> bool:
    """
    Fallback logic when AI decision fails.
    Uses keyword-based heuristics to determine if live data is needed.
    """
    prompt_lower = prompt.lower()
    
    # Keywords that typically require live data
    live_data_indicators = [
        'latest', 'recent', 'new', 'current', 'today', 'now', 'this week', 'this month',
        'price', 'cost', 'stock', 'market', 'weather', 'news', 'announced', 'released',
        'update', 'status', 'available', 'live', 'real-time', 'current', 'happening',
        'trending', 'viral', 'breaking', 'just', 'yesterday', 'last week'
    ]
    
    # Check for live data indicators
    live_data_score = sum(1 for indicator in live_data_indicators if indicator in prompt_lower)
    
    # Keywords that typically don't need live data
    general_knowledge_indicators = [
        'what is', 'how to', 'explain', 'define', 'meaning', 'definition', 'history',
        'theory', 'concept', 'learn', 'understand', 'difference between', 'compare',
        'advantages', 'disadvantages', 'benefits', 'how does', 'why does'
    ]
    
    general_knowledge_score = sum(1 for indicator in general_knowledge_indicators if indicator in prompt_lower)
    
    # Decision logic
    if live_data_score > general_knowledge_score:
        print(f"[DEBUG] Fallback decision: Live data needed (score: {live_data_score} vs {general_knowledge_score})")
        return True
    elif general_knowledge_score > live_data_score:
        print(f"[DEBUG] Fallback decision: No live data needed (score: {general_knowledge_score} vs {live_data_score})")
        return False
    else:
        # If scores are equal or both zero, default to needing live data for safety
        print("[DEBUG] Fallback decision: Defaulting to live data for safety")
        return True
