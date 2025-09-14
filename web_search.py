"""
web_search.py
Web search and browsing functions (duckduckgo, readability...)
"""
from ddgs import DDGS
import requests
from bs4 import BeautifulSoup
from readability import Document
import random

import conf_module

def browse(query: str, num_results: int = 5) -> str:
    """
    Browse the web or search using DuckDuckGo.

    Args:
        query (str): The search query or URL to browse.
        num_results (int): Number of search results to return if using DuckDuckGo.
    
    Returns:
        str: The main text of the webpage if a URL is provided, otherwise search results.
    """
    query = query.strip()

    if query.lower().startswith(("http://", "https://")):
        try:
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
            response = requests.get(query, headers=headers, timeout=10)
            response.raise_for_status()

            # Use Readability to extract the main article
            doc = Document(response.text)
            html = doc.summary()

            soup = BeautifulSoup(html, 'html.parser')
            text = soup.get_text(separator='\n', strip=True)
            return text[:1000]

        except Exception as e:
            return f"Error: {e}"

    else:
        results = DDGS().text(query, max_results=num_results)
        return {"query": query, "results": results}

def gif(query: str) -> str:
    """
    Searches for a GIF on Tenor and returns a random URL from the first 5 results.
    
    Args:
        query (str): The search query for the GIF.
    
    Returns:
        str: URL of a random GIF from the search results or an error message.
    """
    API_KEY = conf_module.load_conf('GIF_TOKEN')
    url = "https://tenor.googleapis.com/v2/search"
    
    params = {
        "q": query,
        "key": API_KEY,
        "limit": 5,
        "media_filter": "minimal"
    }

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()

        if "results" in data and len(data["results"]) > 0:
            top_five_results = data["results"][:5]  # Get only the first 5 results
            random_gif = random.choice(top_five_results)  # Select a random GIF from these
            return random_gif["media_formats"]["gif"]["url"]

        return "No GIF found."

    except requests.exceptions.RequestException as e:
        return f"Error: {e}"