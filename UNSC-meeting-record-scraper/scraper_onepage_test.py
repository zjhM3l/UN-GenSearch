# -*- coding: utf-8 -*-
import sys
import io
import os
import time
import random
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Fix Windows terminal encoding issues
if sys.stdout.encoding != 'UTF-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# ----------------- Configuration -----------------
BASE_URL = "https://digitallibrary.un.org"
VOTING_PAGE_URL_TEMPLATE = "https://digitallibrary.un.org/search?cc=Voting+Data&c=Voting+Data&ln=en&fct__2=Security+Council&jrec={}&rg=50"
DOWNLOAD_DIR = "./UN_Voting_PDFs"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": BASE_URL,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8"
}

# ----------------- Debugging Configuration -----------------
DEBUG_MODE = True  # Enable debugging mode
DEBUG_DIR = "./debug_html"
os.makedirs(DEBUG_DIR, exist_ok=True)

# ----------------- Retry Configuration -----------------
retry_strategy = Retry(
    total=3,
    backoff_factor=1,
    status_forcelist=[429, 500, 502, 503, 504]
)
adapter = HTTPAdapter(max_retries=retry_strategy)
session = requests.Session()
session.mount("https://", adapter)

def save_debug_html(content, filename):
    """
    Save the HTML content for debugging purposes.
    :param content: HTML content to save
    :param filename: Name of the debug file
    """
    path = os.path.join(DEBUG_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Debug file saved: {path}")

def get_safe_soup(url):
    """
    Fetch and parse the HTML content of a given URL.
    :param url: URL to fetch
    :return: BeautifulSoup object or None if an error occurs
    """
    try:
        response = session.get(url, headers=HEADERS, timeout=15)
        response.raise_for_status()
        
        # Save raw HTML for debugging
        if DEBUG_MODE:
            save_debug_html(response.text, f"raw_{url.split('/')[-1]}.html")
            
        return BeautifulSoup(response.text, "html.parser")
    except Exception as e:
        print(f"Request failed: {url} - {str(e)}")
        return None

def get_meeting_record(voting_url):
    """
    Extract the meeting record link from the voting page.
    :param voting_url: URL of the voting page
    :return: URL of the meeting record or None if not found
    """
    soup = get_safe_soup(voting_url)
    if not soup:
        return None

    print("\n=== Updated Locator Logic ===")
    
    # Debugging: Print page title
    title_tag = soup.find("title")
    print(f"Page Title: {title_tag.text if title_tag else 'No Title'}")

    # Debugging: List all custom components
    custom_components = soup.find_all(lambda tag: tag.name.startswith("tindui-"))
    print(f"\nFound {len(custom_components)} custom components:")
    for comp in custom_components:
        print(f"Component Name: {comp.name}")
        print(f"Attributes: {comp.attrs}")
        print("-" * 50)

    # Debugging: Preview HTML structure
    print("\nHTML Structure Preview:")
    print(soup.prettify()[:1000])
    
    # Locate all download components
    download_components = soup.find_all("tindui-app-file-download-link")
    print(f"\nFound {len(download_components)} download components")

    # If no components are found, use an alternative method
    if not download_components:
        print("\n!!! Alternative Method: Searching raw HTML for links !!!")
        all_links = soup.find_all("a", href=True)
        en_pdf_links = [link["href"] for link in all_links if "-EN.pdf" in link["href"]]
        print(f"Found {len(en_pdf_links)} suspected English PDF links")
        if en_pdf_links:
            full_url = BASE_URL + en_pdf_links[0]
            print(f"Alternative link: {full_url}")
            return full_url

    # Find English PDF link
    for component in download_components:
        url = component.get("url", "")
        if "-EN.pdf" in url:
            full_url = BASE_URL + url
            print(f"Found English PDF link: {full_url}")
            return full_url

    print("!!! No English PDF component found !!!")
    return None

def download_file(url, filename):
    """
    Download a file and save it locally.
    :param url: URL of the file to download
    :param filename: Name of the file to save
    :return: True if successful, False otherwise
    """
    try:
        filepath = os.path.join(DOWNLOAD_DIR, filename)
        if os.path.exists(filepath):
            print(f"File already exists: {filename}")
            return True

        headers = {
            **HEADERS,
            "Referer": url  # Essential: Include the referer header
        }

        with session.get(url, headers=headers, stream=True, timeout=30) as r:
            r.raise_for_status()
            total_size = int(r.headers.get('content-length', 0))
            
            with open(filepath, "wb") as f:
                with tqdm(
                    total=total_size, unit='B',
                    unit_scale=True, desc=filename[:20],
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
                ) as pbar:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
                        pbar.update(len(chunk))
        return True
    except Exception as e:
        print(f"Download failed: {str(e)}")
        return False

def main():
    """
    Main function: Process voting records, fetch meeting records, and download PDFs.
    """
    TEST_MODE = True
    records = ["https://digitallibrary.un.org/record/4031209"]
    
    print(f"Total records to process: {len(records)}")

    for idx, voting_url in enumerate(records):
        print(f"\n=== Processing record {idx+1}/{len(records)} ===")
        print(f"Voting Page: {voting_url}")

        # Fetch PDF link
        pdf_url = get_meeting_record(voting_url)
        print(f"PDF Link: {pdf_url}" if pdf_url else "No PDF link found")

        if not pdf_url:
            continue

        # Extract filename from URL
        filename = pdf_url.split("/")[-1]
        
        print(f"Starting download: {filename}")
        if download_file(pdf_url, filename):
            print(f"Download successful: {filename}")
        else:
            print("Download failed")

if __name__ == "__main__":
    main()
