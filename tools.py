"""
Tools available to the research agent.

Each tool is a plain Python function decorated with @tool from LangChain.
The docstring IS the tool description the LLM reads to decide when to use it —
keep it precise, since a vague docstring leads to the agent misusing the tool.

Failure handling philosophy: network calls WILL occasionally time out, get
rate-limited, or hit a dead/blocked URL. Rather than let a single flaky
request burn one of the agent's limited iterations on a dead end, these
tools retry transient failures automatically and return a clear, specific
error message on anything that isn't worth retrying — so the agent (which
is told never to retry the same URL) can make a good decision about what
to do next instead of just seeing an opaque exception string.

Search provider note: web_search uses DuckDuckGo (via the `ddgs` package),
which requires no API key and has no paid quota to run out of — useful for
free development and for anyone cloning this repo without needing to sign
up for anything. If you have a SerpAPI key and want Google-quality results
instead, see the commented-out SerpAPI implementation below web_search for
an easy swap.
"""

import os
import re
import time
from urllib.parse import urlparse
from urllib import robotparser

import requests
from bs4 import BeautifulSoup
from langchain.tools import tool
from ddgs import DDGS
from ddgs.exceptions import DDGSException, RatelimitException, TimeoutException

SERPAPI_KEY = os.getenv("SERPAPI_API_KEY")

# --- Shared retry/failure-handling helpers -------------------------------

MAX_RETRIES = 1                 # total attempts = MAX_RETRIES + 1
RETRY_BACKOFF_SECONDS = 1.5
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}  # transient — worth a retry
PAYWALL_MARKERS = (
    "subscribe to continue",
    "subscribe to read",
    "create a free account to continue",
    "sign in to continue reading",
    "this content is for subscribers",
)

_robots_cache: dict = {}  # per-domain RobotFileParser, cached for process lifetime


def _request_with_retry(method: str, url: str, max_retries: int = MAX_RETRIES, **kwargs):
    """Shared retry wrapper. Retries on timeouts, connection errors, and
    transient 5xx/429 responses with short backoff. Does NOT retry on other
    4xx errors (404, 403, etc.) since those won't change on a retry."""
    attempt = 0
    while True:
        attempt += 1
        try:
            resp = requests.request(method, url, **kwargs)
        except (requests.Timeout, requests.ConnectionError):
            if attempt > max_retries:
                raise
            time.sleep(RETRY_BACKOFF_SECONDS * attempt)
            continue

        if resp.status_code in RETRYABLE_STATUS_CODES and attempt <= max_retries:
            time.sleep(RETRY_BACKOFF_SECONDS * attempt)
            continue
        return resp


def _is_scraping_allowed(url: str) -> bool:
    """Checks robots.txt for the target site before scraping, caching the
    parsed robots.txt per domain. Fails OPEN (allows scraping) if
    robots.txt can't be fetched or parsed — most sites don't have one, and
    treating "unknown" as "disallowed" would block huge amounts of
    legitimate, scrapeable content."""
    try:
        parsed = urlparse(url)
        domain = f"{parsed.scheme}://{parsed.netloc}"
    except Exception:
        return True

    if domain not in _robots_cache:
        rp = robotparser.RobotFileParser()
        rp.set_url(f"{domain}/robots.txt")
        try:
            rp.read()
            _robots_cache[domain] = rp
        except Exception:
            _robots_cache[domain] = None  # unknown -> fail open

    rp = _robots_cache[domain]
    if rp is None:
        return True
    try:
        return rp.can_fetch("ResearchAgent/1.0", url)
    except Exception:
        return True


# --- Tools -----------------------------------------------------------------

@tool
def web_search(query: str) -> str:
    """Searches the web (via DuckDuckGo) for the given query and returns the
    top results as title/link/snippet text. Use this first to find
    information or candidate URLs before deciding whether to scrape a full
    page. Automatically retries once on a rate limit or timeout."""
    max_retries = MAX_RETRIES
    attempt = 0
    last_error = None

    while attempt <= max_retries:
        attempt += 1
        try:
            with DDGS() as ddgs:
                raw_results = list(ddgs.text(query, max_results=5))
            break
        except RatelimitException as e:
            last_error = e
            if attempt <= max_retries:
                time.sleep(RETRY_BACKOFF_SECONDS * attempt)
                continue
            return (f"Search failed: rate-limited after {max_retries + 1} attempt(s). "
                     f"Wait a moment before searching again, or try a different query.")
        except TimeoutException as e:
            last_error = e
            if attempt <= max_retries:
                time.sleep(RETRY_BACKOFF_SECONDS * attempt)
                continue
            return f"Search failed: timed out after {max_retries + 1} attempt(s). Try a different query."
        except DDGSException as e:
            return f"Search failed: {e}"
        except Exception as e:
            return f"Search failed: {e}"
    else:
        return f"Search failed after {max_retries + 1} attempt(s): {last_error}"

    if not raw_results:
        return "No results found for that query. Try rephrasing with different or broader keywords."

    results = []
    for item in raw_results:
        title = item.get("title", "")
        link = item.get("href", "")
        snippet = item.get("body", "")
        results.append(f"- {title}\n  {link}\n  {snippet}")

    return "\n".join(results)


# --- Alternative: SerpAPI (Google results) ---------------------------------
# Swap this in instead of the DuckDuckGo version above if you have a SerpAPI
# key and want Google-quality results / are past your free monthly quota
# concerns. Requires SERPAPI_API_KEY in .env. To use: rename this function
# to web_search and rename/remove the DuckDuckGo version above (only one
# function named web_search can be registered as a tool at a time).
#
# @tool
# def web_search_serpapi(query: str) -> str:
#     """Searches Google via SerpAPI for the given query and returns the top
#     results as title/link/snippet text."""
#     serpapi_key = os.getenv("SERPAPI_API_KEY")
#     if not serpapi_key:
#         return "Error: SERPAPI_API_KEY is not set in the environment."
#
#     params = {"q": query, "api_key": serpapi_key, "engine": "google"}
#     try:
#         resp = _request_with_retry("GET", "https://serpapi.com/search", params=params, timeout=15)
#         resp.raise_for_status()
#         data = resp.json()
#     except requests.Timeout:
#         return (f"Search failed: timed out after {MAX_RETRIES + 1} attempt(s). "
#                 f"Try a shorter or more specific query.")
#     except requests.ConnectionError as e:
#         return f"Search failed: connection error after {MAX_RETRIES + 1} attempt(s) ({e})."
#     except requests.HTTPError as e:
#         return f"Search failed: SerpAPI returned an error ({e})."
#     except Exception as e:
#         return f"Search failed: {e}"
#
#     results = []
#     for item in data.get("organic_results", [])[:5]:
#         title = item.get("title", "")
#         link = item.get("link", "")
#         snippet = item.get("snippet", "")
#         results.append(f"- {title}\n  {link}\n  {snippet}")
#
#     if not results:
#         return "No results found for that query. Try rephrasing with different or broader keywords."
#
#     return "\n".join(results)


@tool
def scrape_page(url: str) -> str:
    """Fetches a webpage and returns its main readable text content with HTML
    tags stripped, truncated to ~3000 characters. Automatically retries once
    on a timeout or transient server error, and respects robots.txt. Use
    sparingly — only when a search snippet doesn't already answer the
    question. If this returns an error or a "skipped"/"blocked" message,
    do NOT retry the same URL — use a different source instead."""
    if not _is_scraping_allowed(url):
        return ("Skipped: this site's robots.txt disallows automated access to this page. "
                "Use a different source instead.")

    try:
        resp = _request_with_retry(
            "GET", url, timeout=15,
            headers={"User-Agent": "Mozilla/5.0 (compatible; ResearchAgent/1.0)"},
        )
    except requests.Timeout:
        return (f"Failed to fetch page: timed out after {MAX_RETRIES + 1} attempt(s). "
                f"This site may be slow or unreachable — use a different source.")
    except requests.ConnectionError as e:
        return (f"Failed to fetch page: connection error after {MAX_RETRIES + 1} attempt(s) ({e}). "
                f"Use a different source.")
    except Exception as e:
        return f"Failed to fetch page: {e}"

    if resp.status_code == 404:
        return ("Failed to fetch page: 404 Not Found. This URL does not exist — "
                "do not guess at similar URLs, run a new web_search instead.")
    if resp.status_code in (401, 403):
        return (f"Failed to fetch page: {resp.status_code} Forbidden. This site is blocking "
                f"automated access (likely bot protection or a login wall). Use a different source.")
    if resp.status_code == 429:
        return "Failed to fetch page: rate-limited (429) even after a retry. Use a different source."
    if resp.status_code >= 500:
        return (f"Failed to fetch page: server error ({resp.status_code}) even after a retry. "
                f"This site may be temporarily down — use a different source.")
    if not resp.ok:
        return f"Failed to fetch page: HTTP {resp.status_code}."

    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "noscript"]):
        tag.decompose()
    text = " ".join(soup.get_text(separator=" ").split())

    if not text:
        return ("Page returned no extractable text content — it may be a JS-rendered page "
                "this tool can't read. Use a different source.")

    lowered = text.lower()
    if len(text) < 400 and any(marker in lowered for marker in PAYWALL_MARKERS):
        return ("Page appears to be behind a paywall or login wall — only a short preview "
                "was extractable. Use a different source.")

    return text[:3000]


@tool
def calculator(expression: str) -> str:
    """Evaluates a basic arithmetic expression, e.g. '19*12 - 24.99*12' or
    '(50-40)/50*100'. Use this for comparing prices, totals, or percentages
    rather than doing math yourself."""
    if not re.fullmatch(r"[0-9+\-*/(). %]+", expression):
        return ("Error: expression contains disallowed characters. Pass a plain numeric "
                "expression only — no letters, currency symbols, or units.")
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception as e:
        return f"Calculation error: {e}"