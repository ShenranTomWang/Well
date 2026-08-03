import re, time, requests
from urllib.parse import parse_qs, urlparse, unquote
from requests.exceptions import HTTPError
from bs4 import BeautifulSoup
from typing import Dict, List

WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"
wiki_session = requests.Session()
wiki_session.headers.update({"User-Agent": "cancer-wikipedia-passages-script/1.0"})
web_session = requests.Session()
web_session.headers.update({"User-Agent": "generic-url-passages-script/1.0"})

def chunk_text_to_passages(text: str) -> List[str]:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return [p.strip() for p in text.split("\n\n") if p.strip()]

def _extract_google_result_url(href: str) -> str | None:
    if href.startswith("/url?"):
        query = parse_qs(urlparse(href).query)
        url = query.get("q", [None])[0]
        if url is not None:
            return unquote(url)
    if href.startswith("http://") or href.startswith("https://"):
        return href
    return None

def google_search_urls(query: str, top_k: int = 5) -> List[str]:
    try:
        r = web_session.get(
            "https://www.google.com/search",
            params={"q": query, "num": top_k},
            timeout=30,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (X11; Linux x86_64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/124.0.0.0 Safari/537.36"
                )
            }
        )
        r.raise_for_status()
    except HTTPError as err:
        print(f"HTTP error for Google search query {query!r}: {err}")
        if err.response.status_code == 429:
            print("Rate limit hit, sleeping for 60 seconds before retrying...")
            time.sleep(60)
        return []

    soup = BeautifulSoup(r.text, "html.parser")
    urls = []
    seen = set()
    for anchor in soup.select("a[href]"):
        url = _extract_google_result_url(anchor["href"])
        if url is None:
            continue
        parsed = urlparse(url)
        if parsed.netloc.endswith("google.com"):
            continue
        if url in seen:
            continue
        seen.add(url)
        urls.append(url)
        if len(urls) >= top_k:
            break
    return urls

def get_wikipedia_passages(cancer: str, cache: Dict[str, List[str]] = {}):
    if cancer in cache:
        print(f"{cancer}: cached, {len(cache[cancer])} passages")
        return cache[cancer]
    text = ""
    try:
        r = wiki_session.get(
            WIKIPEDIA_API,
            params={
                "action": "query",
                "format": "json",
                "formatversion": 2,
                "list": "search",
                "srsearch": cancer,
                "srlimit": 1,
            },
            timeout=30,
        )
        r.raise_for_status()
        obj = r.json()
        results = obj.get("query", {}).get("search", [])
        if results:
            best_title = results[0]["title"]
            r = wiki_session.get(
                WIKIPEDIA_API,
                params={
                    "action": "query",
                    "format": "json",
                    "formatversion": 2,
                    "redirects": 1,
                    "prop": "extracts",
                    "explaintext": 1,
                    "titles": best_title,
                },
                timeout=30,
            )
            r.raise_for_status()
            obj = r.json()
            pages = obj.get("query", {}).get("pages", [])
            page = pages[0] if pages else {}
            if not page.get("missing"):
                text = page.get("extract", "")
    except HTTPError as err:
        print(f"HTTP error for {cancer}: {err}")
        if err.response.status_code == 429:
            print("Rate limit hit, sleeping for 60 seconds before retrying...")
            time.sleep(60)
    passages = chunk_text_to_passages(text)
    cache[cancer] = passages
    return passages

def scrape_text(url: str):
    r = web_session.get(url, timeout=30)
    r.raise_for_status()
    html = r.text
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    return text

def get_url_passages(url: str):
    text = ""
    try:
        text = scrape_text(url)
    except HTTPError as err:
        print(f"HTTP error for {url}: {err}")
        if err.response.status_code == 429:
            print("Rate limit hit, sleeping for 60 seconds before retrying...")
            time.sleep(60)
            text = get_url_passages(url)
    return chunk_text_to_passages(text)

def get_question_search_passages(question: str, top_k: int = 5) -> List[str]:
    passages = []
    for url in google_search_urls(question, top_k=top_k):
        try:
            passages.extend(get_url_passages(url))
        except Exception as err:
            print(f"Question search result {url} error: {err}")
    return passages
