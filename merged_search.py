import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from ddgs import DDGS
import requests
from requests.exceptions import RequestException, Timeout
import trafilatura
from urllib.parse import urlparse

load_dotenv()

client = InferenceClient(
    provider="publicai",
    api_key=os.getenv("HF_TOKEN"),
)

prompt_template = """
Generate a search query for the following user prompt. The query should be concise and relevant to the topic and suitable for search engines.
---

User prompt: 
"""

def generate_search_query(user_prompt: str) -> str:
    prompt = prompt_template + user_prompt
    completion = client.chat.completions.create(
        model="swiss-ai/Apertus-70B-Instruct-2509",
        messages=[
            {"role": "user", "content": prompt}
        ],
    )
    return completion.choices[0].message.content



DEFAULT_UA = "Mozilla/5.0 (compatible; InfoHarvester/1.0)"

def _search_urls(query: str, max_results: int):
    with DDGS() as ddgs:
        # ddgs.text returns dicts with "href"/"link" and "title"
        for r in ddgs.text(query, max_results=max_results, safesearch="Off"):
            u = r.get("href") or r.get("link")
            t = r.get("title")
            if u:
                yield (u, t)

def _extract_paragraphs_from_html(html: str, min_paragraph_len: int):
    extracted = trafilatura.extract(html, include_comments=False, include_tables=False)
    if not extracted:
        return []
    return [p.strip() for p in extracted.split("\n") if p.strip() and len(p.strip()) >= min_paragraph_len]

def gather_paragraphs(
    query: str,
    max_results: int = 8,
    min_paragraph_len: int = 200,
    timeout: float = 12.0,
    user_agent: str = DEFAULT_UA,
) -> list[tuple[str, str]]:
    """
    Search for pages matching `query`, extract paragraphs, and return a list of
    (paragraph, source_url) tuples. Tries direct HTML parse first, then falls
    back to trafilatura.fetch_url if needed.
    """
    session = requests.Session()
    session.headers.update({
        "User-Agent": user_agent,
        "Accept": "text/html,application/xhtml+xml",
    })

    results: list[tuple[str, str]] = []

    for url, _title in _search_urls(query, max_results):
        try:
            resp = session.get(url, timeout=timeout)
            content_type = resp.headers.get("Content-Type", "") or ""
            paras = []

            # Primary path: parse HTML we fetched
            if "text/html" in content_type and resp.text:
                paras = _extract_paragraphs_from_html(resp.text, min_paragraph_len) or []

            # Fallback: let trafilatura handle fetching/decoding
            if not paras:
                downloaded = trafilatura.fetch_url(url)
                if downloaded:
                    paras = _extract_paragraphs_from_html(downloaded, min_paragraph_len) or []

            # Append as (paragraph, url)
            if paras:
                results.extend((para, url) for para in paras)

        except (RequestException, Timeout):
            continue

    return results


def gather_paragraphs_with_sources(query: str,
                                   max_results: int = 8,
                                   min_paragraph_len: int = 200,
                                   timeout: float = 12.0,
                                   user_agent: str = DEFAULT_UA):
    session = requests.Session()
    session.headers.update({"User-Agent": user_agent, "Accept": "text/html,application/xhtml+xml"})
    results = []

    for url, title in _search_urls(query, max_results):
        try:
            resp = session.get(url, timeout=timeout)
            if "text/html" not in (resp.headers.get("Content-Type", "") or ""):
                continue

            paras = _extract_paragraphs_from_html(resp.text, min_paragraph_len)
            if not paras:
                downloaded = trafilatura.fetch_url(url)
                if downloaded:
                    paras = _extract_paragraphs_from_html(downloaded, min_paragraph_len)

            if paras:
                results.append({
                    "url": url,
                    "title": title or (urlparse(url).netloc if url else None),
                    "paragraphs": paras
                })
        except (RequestException, Timeout):
            continue

    return results

# Takes user prompt and returns paragraphs on the topic

def merged_search(user_prompt: str):
    query = generate_search_query(user_prompt)
    paragraphs = gather_paragraphs(query)
    return paragraphs


### Example usage 

if __name__ == "__main__":
    paragraphs = merged_search("What tech products were announced at the most recent Apple event?")
    for p in paragraphs:
        print(p)
        print("---")
        print(len(p))
