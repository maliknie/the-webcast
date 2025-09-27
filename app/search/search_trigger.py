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
    global client
    
    # Initialize client if not already done
    if client is None:
        api_key = os.getenv("SWISS_AI_PLATFORM_API_KEY")
        if not api_key:
            print("[DEBUG] SWISS_AI_PLATFORM_API_KEY not found, defaulting to True")
            return True
        client = openai.OpenAI(
            api_key=api_key,
            base_url="https://api.swisscom.com/layer/swiss-ai-weeks/apertus-70b/v1"
        )
    
    system_message = (
        "If you cannot answer to the user prompt -- whether it's a question or not -- with the informations that you have, reply with 'no'. "
        "If you can answer with your existing knowledge, reply with 'yes'. "
        "Respond with exactly one word: 'yes' or 'no'. Do not add punctuation or explanation."
    )

    try:
        print(f"[DEBUG] Sending prompt: {prompt}")

        response = client.chat.completions.create(
            model="swiss-ai/Apertus-70B",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ],
            logit_bias = logit_bias,
            temperature = 0.1
        )

        choices = response.to_dict().get("choices", [])

        answer = choices[0]["message"]["content"].strip().lower()
        print(answer)
        first_word = answer.split()[0]
        if first_word not in ["yes", "no"]:
            first_word = "yes" if "yes" in answer else "no"

        return first_word == "no"

    except Exception as e:
        print(f"[DEBUG] Exception occurred: {e}")
        return True
