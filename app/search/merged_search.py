import os
import threading
import time
import re
import math
from collections import Counter
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
    api_key=os.getenv("HG_API_KEY"),
)

prompt_template = """
Generate a highly effective search query for the following user prompt. The query should:
1. Be concise (5-8 words maximum)
2. Include specific, searchable keywords and proper nouns
3. Use terms that are likely to appear in authoritative sources
4. Focus on factual, verifiable information
5. Include relevant dates, names, or specific terms when applicable
6. For "latest" or "recent" queries, include current year (2025) and official terms
7. Use industry-standard terminology and official names
8. Be optimized for finding reliable, factual content from official sources
9. For Apple/iPhone queries, prioritize official Apple announcements and tech news

Examples of good queries:
- "iPhone 17 Pro 2025 specifications features" (for latest iPhone info)
- "Apple iPhone 2025 release date" (for recent releases)
- "OpenAI GPT-5 capabilities performance" (instead of "AI latest developments")

User prompt: 
"""

def generate_search_query(user_prompt: str) -> str:
    """
    Generate an optimized search query with fallback strategies for reliability.
    """
    try:
        prompt = prompt_template + user_prompt
        completion = client.chat.completions.create(
            model="swiss-ai/Apertus-70B-Instruct-2509",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,  # Lower temperature for more consistent results
            max_tokens=50    # Limit response length
        )
        
        query = completion.choices[0].message.content.strip()
        
        # Validate and clean the query
        if query and len(query.split()) <= 10:  # Ensure it's concise
            return query
        else:
            print(f"[DEBUG] Generated query too long or empty, using fallback")
            return _generate_fallback_query(user_prompt)
            
    except Exception as e:
        print(f"[DEBUG] Error generating search query: {e}")
        return _generate_fallback_query(user_prompt)

def _generate_fallback_query(user_prompt: str) -> str:
    """
    Generate a simple fallback query when the AI query generation fails.
    """
    # Extract key terms and create a simple search query
    words = user_prompt.lower().split()
    
    # Remove common stop words
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'what', 'when', 'where', 'why', 'how', 'who', 'latest', 'recent', 'new'}
    
    # Keep important words (nouns, proper nouns, specific terms)
    important_words = [word for word in words if word not in stop_words and len(word) > 2]
    
    # Take the most important words (up to 6)
    fallback_query = ' '.join(important_words[:6])
    
    if not fallback_query:
        # Ultimate fallback - use the original prompt
        fallback_query = user_prompt[:50]  # Truncate if too long
    
    print(f"[DEBUG] Using fallback query: {fallback_query}")
    return fallback_query



DEFAULT_UA = "Mozilla/5.0 (compatible; InfoHarvester/1.0)"

# Domains to exclude from search results (opinion-based or unreliable sources)
EXCLUDED_DOMAINS = {
    'reddit.com', 'www.reddit.com', 'old.reddit.com',
    'ask.com', 'www.ask.com', 
    'instagram.com', 'tiktok.com', 'tumblr.com',
    'huffpost.com', 'support.apple.com', 'appleinsider.com',
}

# Preferred domains for authoritative content
PREFERRED_DOMAINS = {
    'wikipedia.org', 'reuters.com', 'ap.org', 'bbc.com', 'cnn.com',
    'nytimes.com', 'washingtonpost.com', 'wsj.com',
    'bloomberg.com', 'forbes.com', 'techcrunch.com',
    'theverge.com', 'arstechnica.com', 'wired.com',
    'google.com',
    'nasa.gov', 'nih.gov', 'cdc.gov', 'who.int',
    'nature.com', 'science.org', 'cell.com',
    'ieee.org', 'acm.org', 'springer.com', 
    'google.com',
    # Apple and tech-specific authoritative sources
    'apple.com', 'www.apple.com',
    'macrumors.com', '9to5mac.com', 
    'engadget.com', 'mashable.com', 'cnet.com',
    'gsmarena.com', 'phonearena.com', 'pocket-lint.com',
    'digitaltrends.com', 'slashgear.com', 'bgr.com',
    'imore.com', 'cultofmac.com', 'macworld.com'
}

def _is_excluded_domain(url: str) -> bool:
    """
    Check if a URL should be excluded based on domain filtering.
    """
    try:
        from urllib.parse import urlparse
        parsed_url = urlparse(url)
        domain = parsed_url.netloc.lower()
        
        # Remove 'www.' prefix for comparison
        if domain.startswith('www.'):
            domain = domain[4:]
        
        # Check if domain is in excluded list
        if domain in EXCLUDED_DOMAINS:
            print(f"[DEBUG] Excluding domain: {domain}")
            return True
        
        # Check for subdomains of excluded domains
        for excluded_domain in EXCLUDED_DOMAINS:
            if domain.endswith('.' + excluded_domain):
                print(f"[DEBUG] Excluding subdomain: {domain}")
                return True
        
        return False
        
    except Exception as e:
        print(f"[DEBUG] Error checking domain exclusion: {e}")
        return False

def _is_preferred_domain(url: str) -> bool:
    """
    Check if a URL is from a preferred authoritative domain.
    """
    try:
        from urllib.parse import urlparse
        parsed_url = urlparse(url)
        domain = parsed_url.netloc.lower()
        
        # Remove 'www.' prefix for comparison
        if domain.startswith('www.'):
            domain = domain[4:]
        
        # Check if domain is in preferred list
        if domain in PREFERRED_DOMAINS:
            return True
        
        # Check for subdomains of preferred domains
        for preferred_domain in PREFERRED_DOMAINS:
            if domain.endswith('.' + preferred_domain):
                return True
        
        return False
        
    except Exception as e:
        print(f"[DEBUG] Error checking preferred domain: {e}")
        return False

# Text preprocessing for cosine similarity
def _preprocess_text(text: str) -> list[str]:
    """
    Preprocess text for cosine similarity calculation.
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation and special characters
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # Split into words and remove empty strings
    words = [word for word in text.split() if word]
    
    # Remove common stop words
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
        'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did',
        'will', 'would', 'could', 'should', 'may', 'might', 'can', 'what', 'when', 'where',
        'why', 'how', 'who', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it',
        'we', 'they', 'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his', 'her', 'its',
        'our', 'their', 'mine', 'yours', 'hers', 'ours', 'theirs'
    }
    
    # Filter out stop words and short words
    filtered_words = [word for word in words if word not in stop_words and len(word) > 2]
    
    return filtered_words

def _get_word_frequency(text: str) -> Counter:
    """
    Get word frequency counter for a text.
    """
    words = _preprocess_text(text)
    return Counter(words)

def _cosine_similarity(text1: str, text2: str) -> float:
    """
    Calculate cosine similarity between two texts.
    Returns a value between 0 and 1, where 1 means identical.
    """
    try:
        # Get word frequencies
        freq1 = _get_word_frequency(text1)
        freq2 = _get_word_frequency(text2)
        
        # Get all unique words from both texts
        all_words = set(freq1.keys()) | set(freq2.keys())
        
        if not all_words:
            return 0.0
        
        # Create vectors
        vector1 = [freq1.get(word, 0) for word in all_words]
        vector2 = [freq2.get(word, 0) for word in all_words]
        
        # Calculate dot product
        dot_product = sum(a * b for a, b in zip(vector1, vector2))
        
        # Calculate magnitudes
        magnitude1 = math.sqrt(sum(a * a for a in vector1))
        magnitude2 = math.sqrt(sum(b * b for b in vector2))
        
        # Avoid division by zero
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        
        # Calculate cosine similarity
        similarity = dot_product / (magnitude1 * magnitude2)
        
        return similarity
        
    except Exception as e:
        print(f"[DEBUG] Error calculating cosine similarity: {e}")
        return 0.0

def _semantic_relevance_score(content: str, query: str) -> float:
    """
    Calculate semantic relevance score using cosine similarity.
    Returns a score between 0 and 1.
    """
    try:
        # Calculate cosine similarity
        similarity = _cosine_similarity(content, query)
        
        # Boost score for exact phrase matches
        content_lower = content.lower()
        query_lower = query.lower()
        
        phrase_boost = 0.0
        if query_lower in content_lower:
            phrase_boost = 0.2  # 20% boost for exact phrase match
        
        # Boost score for important terms (prioritize recent and official info)
        important_terms = ['official', 'confirmed', 'announced', 'released', 'specifications', 
                          'features', 'performance', 'review', 'analysis', 'report', '2024', '2025',
                          'launch', 'unveiled', 'introduced', 'debut', 'premiere']
        term_boost = 0.0
        for term in important_terms:
            if term in content_lower:
                term_boost += 0.05  # 5% boost per important term
        
        # Combine scores
        final_score = min(1.0, similarity + phrase_boost + term_boost)
        
        return final_score
        
    except Exception as e:
        print(f"[DEBUG] Error calculating semantic relevance: {e}")
        return 0.0

def _search_urls(query: str, max_results: int):
    """
    Enhanced search with multiple engines, fallback strategies, and domain filtering.
    """
    search_engines = [
        _search_with_ddg,
        _search_with_alternative
    ]
    
    all_results = []
    seen_urls = set()
    excluded_count = 0
    
    for search_func in search_engines:
        try:
            results = list(search_func(query, max_results * 2))  # Get more results to account for filtering
            for url, title in results:
                if url not in seen_urls:
                    # Check if domain should be excluded
                    if _is_excluded_domain(url):
                        excluded_count += 1
                        continue
                    
                    all_results.append((url, title))
                    seen_urls.add(url)
                    
            # If we have enough results, break
            if len(all_results) >= max_results:
                break
                
        except Exception as e:
            print(f"[DEBUG] Search engine {search_func.__name__} failed: {e}")
            continue
    
    print(f"[DEBUG] Domain filtering: {excluded_count} URLs excluded, {len(all_results)} URLs kept")
    
    # Return up to max_results
    return all_results[:max_results]

def _search_with_ddg(query: str, max_results: int):
    """Search using DuckDuckGo (primary engine)"""
    try:
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results, safesearch="Off"):
                u = r.get("href") or r.get("link")
                t = r.get("title")
                if u:
                    yield (u, t)
    except Exception as e:
        print(f"[DEBUG] DDG search failed: {e}")
        return

def _search_with_alternative(query: str, max_results: int):
    """Alternative search using different parameters"""
    try:
        with DDGS() as ddgs:
            # Try with different search parameters
            for r in ddgs.text(query, max_results=max_results, safesearch="Moderate", region="us-en"):
                u = r.get("href") or r.get("link")
                t = r.get("title")
                if u:
                    yield (u, t)
    except Exception as e:
        print(f"[DEBUG] Alternative search failed: {e}")
        return

def _extract_paragraphs_from_html(html: str, min_paragraph_len: int):
    """
    Extract paragraphs from HTML using multiple methods for better reliability.
    """
    extraction_methods = [
        _extract_with_trafilatura,
        _extract_with_beautifulsoup,
        _extract_with_regex
    ]
    
    for method in extraction_methods:
        try:
            paragraphs = method(html, min_paragraph_len)
            if paragraphs:
                print(f"[DEBUG] Successfully extracted {len(paragraphs)} paragraphs using {method.__name__}")
                return paragraphs
        except Exception as e:
            print(f"[DEBUG] Method {method.__name__} failed: {e}")
            continue
    
    print("[DEBUG] All extraction methods failed")
    return []

def _extract_with_trafilatura(html: str, min_paragraph_len: int):
    """Extract using trafilatura (preferred method)"""
    if threading.current_thread() is threading.main_thread():
        extracted = trafilatura.extract(html, include_comments=False, include_tables=False)
        if extracted:
            return [p.strip() for p in extracted.split("\n") if p.strip() and len(p.strip()) >= min_paragraph_len]
    return []

def _extract_with_beautifulsoup(html: str, min_paragraph_len: int):
    """Extract using BeautifulSoup as fallback"""
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, 'html.parser')
        
        # Remove script, style, and other non-content elements
        for element in soup(["script", "style", "nav", "header", "footer", "aside", "advertisement", "menu", "sidebar"]):
            element.decompose()
        
        # Find main content areas
        main_content = soup.find('main') or soup.find('article') or soup.find('div', class_=['content', 'post', 'entry', 'article-content'])
        if main_content:
            soup = main_content
        
        # Get text and clean it up
        text = soup.get_text()
        
        # Split into paragraphs and filter
        lines = text.split('\n')
        paragraphs = []
        current_paragraph = []
        
        for line in lines:
            line = line.strip()
            if line:
                current_paragraph.append(line)
            else:
                if current_paragraph:
                    paragraph = ' '.join(current_paragraph)
                    if len(paragraph) >= min_paragraph_len:
                        paragraphs.append(paragraph)
                    current_paragraph = []
        
        # Don't forget the last paragraph
        if current_paragraph:
            paragraph = ' '.join(current_paragraph)
            if len(paragraph) >= min_paragraph_len:
                paragraphs.append(paragraph)
        
        return paragraphs
    except Exception as e:
        print(f"[DEBUG] BeautifulSoup extraction failed: {e}")
        return []

def _extract_with_regex(html: str, min_paragraph_len: int):
    """Extract using regex as last resort"""
    import re
    
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', ' ', html)
    
    # Clean up whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Split into sentences and group into paragraphs
    sentences = re.split(r'[.!?]+', text)
    paragraphs = []
    current_paragraph = []
    
    for sentence in sentences:
        sentence = sentence.strip()
        if sentence:
            current_paragraph.append(sentence)
            # If we have enough content, make it a paragraph
            if len(' '.join(current_paragraph)) >= min_paragraph_len:
                paragraphs.append(' '.join(current_paragraph))
                current_paragraph = []
    
    return paragraphs

def _simple_text_extraction(html: str, min_paragraph_len: int):
    """Thread-safe fallback text extraction without trafilatura"""
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, 'html.parser')
        
        # Remove script, style, and other non-content elements
        for element in soup(["script", "style", "nav", "header", "footer", "aside", "advertisement"]):
            element.decompose()
        
        # Find main content areas
        main_content = soup.find('main') or soup.find('article') or soup.find('div', class_=['content', 'post', 'entry'])
        if main_content:
            soup = main_content
        
        # Get text and clean it up
        text = soup.get_text()
        
        # Split into paragraphs and filter
        lines = text.split('\n')
        paragraphs = []
        current_paragraph = []
        
        for line in lines:
            line = line.strip()
            if line:
                current_paragraph.append(line)
            else:
                if current_paragraph:
                    paragraph = ' '.join(current_paragraph)
                    if len(paragraph) >= min_paragraph_len:
                        paragraphs.append(paragraph)
                    current_paragraph = []
        
        # Don't forget the last paragraph
        if current_paragraph:
            paragraph = ' '.join(current_paragraph)
            if len(paragraph) >= min_paragraph_len:
                paragraphs.append(paragraph)
        
        return paragraphs
    except Exception as e:
        print(f"[DEBUG] Error in simple text extraction: {e}")
        return []

def _is_relevant_content(content: str, query: str) -> bool:
    """
    Enhanced relevance checking with cosine similarity and multiple scoring methods.
    """
    content_lower = content.lower()
    query_lower = query.lower()
    
    # Extract key terms from query (remove common words)
    common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'what', 'when', 'where', 'why', 'how', 'who', 'latest', 'recent', 'new', 'news', 'update', 'updates'}
    query_terms = [term for term in query_lower.split() if term not in common_words and len(term) > 2]
    
    if not query_terms:
        return True  # If no specific terms, assume relevant
    
    # Method 1: Cosine similarity (semantic relevance)
    semantic_score = _semantic_relevance_score(content, query)
    semantic_threshold = 0.15  # Minimum semantic similarity threshold
    
    # Method 2: Basic term matching
    basic_score = sum(1 for term in query_terms if term in content_lower)
    basic_relevance = basic_score >= max(1, len(query_terms) * 0.3)
    
    # Method 3: Weighted scoring (longer terms get more weight)
    weighted_score = 0
    for term in query_terms:
        if term in content_lower:
            weighted_score += len(term)  # Longer terms are more important
    
    # Method 4: Phrase matching (exact phrase matches get bonus)
    phrase_bonus = 0
    if len(query_terms) > 1:
        # Check for exact phrase matches
        query_phrase = ' '.join(query_terms)
        if query_phrase in content_lower:
            phrase_bonus = 5
    
    # Method 5: Domain-specific relevance (check for authoritative indicators)
    authority_indicators = ['official', 'announced', 'released', 'confirmed', 'specifications', 'features', 'performance', 'review', 'analysis', 'report', '2024', '2025', 'launch', 'unveiled', 'introduced']
    authority_score = sum(1 for indicator in authority_indicators if indicator in content_lower)
    
    # Combined scoring with cosine similarity
    total_score = basic_score + weighted_score + phrase_bonus + authority_score
    min_required_score = max(2, len(query_terms) * 0.5)
    
    # Check both traditional scoring and semantic similarity
    traditional_relevant = total_score >= min_required_score
    semantic_relevant = semantic_score >= semantic_threshold
    
    # Content is relevant if either method indicates relevance
    is_relevant = traditional_relevant or semantic_relevant
    
    if not is_relevant:
        print(f"[DEBUG] Content filtered out - Traditional: {total_score}/{min_required_score}, Semantic: {semantic_score:.3f}/{semantic_threshold}")
        print(f"[DEBUG] Content preview: {content[:100]}...")
    else:
        print(f"[DEBUG] Content accepted - Traditional: {total_score}/{min_required_score}, Semantic: {semantic_score:.3f}/{semantic_threshold}")
    
    return is_relevant

def gather_paragraphs(query: str,
                      max_results: int = 8,
                      min_paragraph_len: int = 200,
                      timeout: float = 12.0,
                      user_agent: str = DEFAULT_UA,
                      max_retries: int = 2) -> list[str]:
    """
    Gather paragraphs with retry mechanism, better error handling, and domain filtering.
    """
    session = requests.Session()
    session.headers.update({
        "User-Agent": user_agent, 
        "Accept": "text/html,application/xhtml+xml",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate",
        "Connection": "keep-alive"
    })
    
    all_paragraphs: list[str] = []
    successful_urls = 0
    failed_urls = 0
    excluded_urls = 0

    for url, title in _search_urls(query, max_results):
        # Double-check domain exclusion (in case it wasn't caught earlier)
        if _is_excluded_domain(url):
            excluded_urls += 1
            print(f"[DEBUG] Skipping excluded domain: {url}")
            continue
            
        paragraphs_found = False
        
        for attempt in range(max_retries + 1):
            try:
                print(f"[DEBUG] Attempting to fetch {url} (attempt {attempt + 1}/{max_retries + 1})")
                
                resp = session.get(url, timeout=timeout)
                
                # Check content type
                content_type = resp.headers.get("Content-Type", "").lower()
                if "text/html" not in content_type:
                    print(f"[DEBUG] Skipping {url} - not HTML content: {content_type}")
                    break
                
                # Check response status
                if resp.status_code != 200:
                    print(f"[DEBUG] HTTP {resp.status_code} for {url}")
                    if attempt < max_retries:
                        continue
                    else:
                        break
                
                # Extract paragraphs
                paras = _extract_paragraphs_from_html(resp.text, min_paragraph_len)
                if not paras:
                    print(f"[DEBUG] No paragraphs extracted from {url}")
                    if attempt < max_retries:
                        continue
                    else:
                        break

                # Filter paragraphs for relevance
                relevant_paras = []
                for para in paras:
                    if _is_relevant_content(para, query):
                        relevant_paras.append(para)
                
                if relevant_paras:
                    all_paragraphs.extend(relevant_paras)
                    print(f"[DEBUG] Added {len(relevant_paras)} relevant paragraphs from {url}")
                    successful_urls += 1
                    paragraphs_found = True
                else:
                    print(f"[DEBUG] No relevant content found in {url}")
                
                break  # Success, no need to retry
                
            except (RequestException, Timeout) as e:
                print(f"[DEBUG] Request error for {url} (attempt {attempt + 1}): {e}")
                if attempt < max_retries:
                    time.sleep(1)  # Wait before retry
                    continue
                else:
                    failed_urls += 1
                    break
            except Exception as e:
                print(f"[DEBUG] Unexpected error for {url}: {e}")
                failed_urls += 1
                break
        
        if not paragraphs_found:
            failed_urls += 1
    
    print(f"[DEBUG] Search completed: {successful_urls} successful, {failed_urls} failed, {excluded_urls} excluded URLs")
    return all_paragraphs

def gather_paragraphs_with_sources(query: str,
                                   max_results: int = 8,
                                   min_paragraph_len: int = 200,
                                   timeout: float = 12.0,
                                   user_agent: str = DEFAULT_UA,
                                   max_retries: int = 2):
    """
    Enhanced version with retry mechanism and better error handling.
    """
    session = requests.Session()
    session.headers.update({
        "User-Agent": user_agent, 
        "Accept": "text/html,application/xhtml+xml",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate",
        "Connection": "keep-alive"
    })
    
    results = []
    successful_urls = 0
    failed_urls = 0

    for url, title in _search_urls(query, max_results):
        paragraphs_found = False
        
        for attempt in range(max_retries + 1):
            try:
                print(f"[DEBUG] Attempting to fetch {url} (attempt {attempt + 1}/{max_retries + 1})")
                
                resp = session.get(url, timeout=timeout)
                
                # Check content type
                content_type = resp.headers.get("Content-Type", "").lower()
                if "text/html" not in content_type:
                    print(f"[DEBUG] Skipping {url} - not HTML content: {content_type}")
                    break
                
                # Check response status
                if resp.status_code != 200:
                    print(f"[DEBUG] HTTP {resp.status_code} for {url}")
                    if attempt < max_retries:
                        continue
                    else:
                        break
                
                # Extract paragraphs
                paras = _extract_paragraphs_from_html(resp.text, min_paragraph_len)
                if not paras:
                    print(f"[DEBUG] No paragraphs extracted from {url}")
                    if attempt < max_retries:
                        continue
                    else:
                        break

                # Filter for relevance
                relevant_paras = []
                for para in paras:
                    if _is_relevant_content(para, query):
                        relevant_paras.append(para)

                if relevant_paras:
                    results.append({
                        "url": url,
                        "title": title or (urlparse(url).netloc if url else None),
                        "paragraphs": relevant_paras
                    })
                    successful_urls += 1
                    paragraphs_found = True
                    print(f"[DEBUG] Added {len(relevant_paras)} relevant paragraphs from {url}")
                else:
                    print(f"[DEBUG] No relevant content found in {url}")
                
                break  # Success, no need to retry
                
            except (RequestException, Timeout) as e:
                print(f"[DEBUG] Request error for {url} (attempt {attempt + 1}): {e}")
                if attempt < max_retries:
                    time.sleep(1)  # Wait before retry
                    continue
                else:
                    failed_urls += 1
                    break
            except Exception as e:
                print(f"[DEBUG] Unexpected error for {url}: {e}")
                failed_urls += 1
                break
        
        if not paragraphs_found:
            failed_urls += 1
    
    print(f"[DEBUG] Search with sources completed: {successful_urls} successful, {failed_urls} failed URLs")
    return results

# Takes user prompt and returns paragraphs on the topic

def merged_search(user_prompt: str):
    """
    Enhanced web search with multiple fallback strategies for maximum reliability.
    """
    try:
        # Step 1: Generate optimized search query
        query = generate_search_query(user_prompt)
        print(f"[DEBUG] Generated search query: {query}")
        
        # Step 2: Try primary search with enhanced parameters
        paragraphs = gather_paragraphs(
            query=query,
            max_results=10,  # Increased for better coverage
            min_paragraph_len=150,  # Slightly lower threshold
            timeout=15.0,  # Longer timeout
            max_retries=3  # More retries
        )
        
        print(f"[DEBUG] Primary search found {len(paragraphs)} paragraphs")
        
        # Step 3: If insufficient results, try alternative search strategies
        if len(paragraphs) < 2:
            print("[DEBUG] Insufficient results, trying alternative search...")
            
            # Try with different query variations
            alternative_queries = _generate_alternative_queries(user_prompt)
            
            for alt_query in alternative_queries[:2]:  # Try up to 2 alternatives
                print(f"[DEBUG] Trying alternative query: {alt_query}")
                alt_paragraphs = gather_paragraphs(
                    query=alt_query,
                    max_results=6,
                    min_paragraph_len=150,
                    timeout=12.0,
                    max_retries=2
                )
                
                if alt_paragraphs:
                    paragraphs.extend(alt_paragraphs)
                    print(f"[DEBUG] Alternative search added {len(alt_paragraphs)} paragraphs")
                    break  # Stop after first successful alternative
        
        # Step 4: Process and rank results
        if paragraphs:
            # Enhanced ranking: combine length, relevance, and quality indicators
            ranked_paragraphs = _rank_paragraphs(paragraphs, query)
            
            # Return top 3-4 paragraphs for better context
            top_paragraphs = ranked_paragraphs[:4]
            print(f"[DEBUG] Returning top {len(top_paragraphs)} ranked paragraphs")
            return top_paragraphs
        else:
            print("[DEBUG] No relevant paragraphs found after all attempts")
            return []
            
    except Exception as e:
        print(f"[DEBUG] Error in merged_search: {e}")
        
        # Handle specific error types gracefully
        if "signal only works in main thread" in str(e) or "signal" in str(e).lower():
            print("[DEBUG] Signal threading error detected - returning empty results")
            return []
        elif "timeout" in str(e).lower():
            print("[DEBUG] Timeout error - returning empty results")
            return []
        elif "connection" in str(e).lower():
            print("[DEBUG] Connection error - returning empty results")
            return []
        else:
            # For other errors, try one more time with basic fallback
            try:
                print("[DEBUG] Attempting fallback search...")
                fallback_query = _generate_fallback_query(user_prompt)
                fallback_paragraphs = gather_paragraphs(
                    query=fallback_query,
                    max_results=5,
                    min_paragraph_len=100,
                    timeout=10.0,
                    max_retries=1
                )
                return fallback_paragraphs[:2] if fallback_paragraphs else []
            except:
                print("[DEBUG] Fallback search also failed")
                return []

def _generate_alternative_queries(user_prompt: str) -> list[str]:
    """Generate alternative search queries for better coverage."""
    alternatives = []
    
    # Extract key terms
    words = user_prompt.lower().split()
    important_words = [w for w in words if len(w) > 3 and w not in {'what', 'when', 'where', 'why', 'how', 'who', 'latest', 'recent', 'new'}]
    
    if len(important_words) >= 2:
        # Create variations with different combinations
        alternatives.append(' '.join(important_words[:3]))
        if len(important_words) > 3:
            alternatives.append(' '.join(important_words[1:4]))
    
    # Add specific domain queries
    if any(word in user_prompt.lower() for word in ['iphone', 'apple', 'ios']):
        alternatives.append('iPhone 15 Pro 2024 specifications features')
        alternatives.append('Apple iPhone latest release 2024')
        alternatives.append('iPhone 15 Pro Max official announcement')
    elif any(word in user_prompt.lower() for word in ['tesla', 'electric', 'car']):
        alternatives.append('Tesla electric vehicle specifications')
    elif any(word in user_prompt.lower() for word in ['ai', 'artificial', 'intelligence']):
        alternatives.append('artificial intelligence developments 2024')
    
    return alternatives

def _rank_paragraphs(paragraphs: list[str], query: str, source_urls: list[str] = None) -> list[str]:
    """Rank paragraphs by relevance and quality using cosine similarity."""
    def paragraph_score(para: str, para_index: int = 0) -> float:
        score = 0
        
        # Length score (prefer medium-length paragraphs)
        length = len(para)
        if 200 <= length <= 800:
            score += 2
        elif 100 <= length < 200 or 800 < length <= 1200:
            score += 1
        
        # Cosine similarity score (semantic relevance) - most important
        semantic_score = _semantic_relevance_score(para, query)
        score += semantic_score * 10  # High weight for semantic similarity
        
        # Traditional relevance score
        query_terms = query.lower().split()
        term_matches = sum(1 for term in query_terms if term in para.lower())
        score += term_matches * 0.5
        
        # Quality indicators (prioritize recent and official info)
        quality_indicators = ['official', 'confirmed', 'announced', 'released', 'specifications', 'features', 'performance', '2024', '2025', 'launch', 'unveiled', 'introduced']
        quality_score = sum(1 for indicator in quality_indicators if indicator in para.lower())
        score += quality_score * 0.3
        
        # Domain authority boost (if source URL is available)
        if source_urls and para_index < len(source_urls):
            source_url = source_urls[para_index]
            if _is_preferred_domain(source_url):
                score += 3  # Significant boost for authoritative domains
                print(f"[DEBUG] Authority boost for {source_url}")
        
        # Avoid low-quality indicators
        low_quality_indicators = ['click here', 'read more', 'advertisement', 'sponsored', 'buy now']
        if any(indicator in para.lower() for indicator in low_quality_indicators):
            score -= 1
        
        return score
    
    # Sort by score (highest first)
    scored_paragraphs = [(para, paragraph_score(para, i)) for i, para in enumerate(paragraphs)]
    scored_paragraphs.sort(key=lambda x: x[1], reverse=True)
    
    # Log top scores for debugging
    print(f"[DEBUG] Top paragraph scores:")
    for i, (para, score) in enumerate(scored_paragraphs[:3]):
        semantic = _semantic_relevance_score(para, query)
        print(f"[DEBUG] #{i+1}: Score={score:.2f}, Semantic={semantic:.3f}, Length={len(para)}")
    
    return [para for para, score in scored_paragraphs]


### Example usage 

if __name__ == "__main__":
    paragraphs = merged_search("What tech products were announced at the most recent Apple event?")
    for p in paragraphs:
        print(p)
