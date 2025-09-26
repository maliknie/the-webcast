from ddgs import DDGS
import requests
from requests.exceptions import RequestException, Timeout
import trafilatura
from urllib.parse import urlparse

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

def gather_paragraphs(query: str,
                      max_results: int = 8,
                      min_paragraph_len: int = 200,
                      timeout: float = 12.0,
                      user_agent: str = DEFAULT_UA) -> list[str]:
    session = requests.Session()
    session.headers.update({"User-Agent": user_agent, "Accept": "text/html,application/xhtml+xml"})
    all_paragraphs: list[str] = []

    for url, _title in _search_urls(query, max_results):
        try:
            resp = session.get(url, timeout=timeout)
            if "text/html" not in (resp.headers.get("Content-Type", "") or ""):
                continue

            paras = _extract_paragraphs_from_html(resp.text, min_paragraph_len)
            if not paras:
                # Let trafilatura handle fetching/encoding itself as a fallback
                downloaded = trafilatura.fetch_url(url)
                if downloaded:
                    paras = _extract_paragraphs_from_html(downloaded, min_paragraph_len)

            all_paragraphs.extend(paras)
        except (RequestException, Timeout):
            continue

    return all_paragraphs

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

rich = gather_paragraphs_with_sources("2024 Presidential election US", max_results=6)
for item in rich:
    print(item["title"], item["url"])
    for p in item["paragraphs"][:2]:
        print()
        print("-", p)
