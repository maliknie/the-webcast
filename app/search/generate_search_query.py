import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

load_dotenv()

client = InferenceClient(
    provider="publicai",
    api_key=os.getenv("HF_API_KEY"),
)

prompt_template = """
Generate a search query for the following user prompt. The query should be concise and relevant to the topic and suitable for search engines.
---

User prompt: 
"""

prompt = prompt_template + "What tech products were announced at the most recent Apple event?"

completion = client.chat.completions.create(
    model="swiss-ai/Apertus-70B-Instruct-2509",
    messages=[
        {"role": "user", "content": prompt}
    ],
)

print(completion.choices[0].message.content)
