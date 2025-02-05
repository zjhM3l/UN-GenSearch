# -*- coding: utf-8 -*-
import os
import time
import random
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import re
import sys

# Fix Windows terminal encoding issue
sys.stdout.reconfigure(encoding='utf-8')

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

# ----------------- Retry Configuration -----------------
retry_strategy = Retry(
    total=3,
    backoff_factor=1,
    status_forcelist=[429, 500, 502, 503, 504]
)
adapter = HTTPAdapter(max_retries=retry_strategy)
session = requests.Session()
session.mount("https://", adapter)

# ----------------- Topics for Scraping -----------------
TOPICS = {
    "Former Yugoslavia Situation": "https://digitallibrary.un.org/search?cc=Voting+Data&ln=en&p=&f=&rm=&sf=&so=d&rg=50&c=Voting+Data&of=hb&fti=0&fct__2=Security+Council&fct__8=FORMER%20YUGOSLAVIA%20SITUATION&fti=0",
    "Bosnia and Herzegovina Situation": "https://digitallibrary.un.org/search?cc=Voting+Data&ln=en&p=&f=&rm=&sf=&so=d&rg=50&c=Voting+Data&of=hb&fti=0&fct__2=Security+Council&fct__8=BOSNIA%20AND%20HERZEGOVINA%20SITUATION&fti=0",
    "Middle East Situation": "https://digitallibrary.un.org/search?cc=Voting+Data&ln=en&p=&f=&rm=&sf=&so=d&rg=50&c=Voting+Data&of=hb&fti=0&fct__2=Security+Council&fct__8=MIDDLE%20EAST%20SITUATION&fti=0",
    "Somalia Situation": "https://digitallibrary.un.org/search?cc=Voting+Data&ln=en&p=&f=&rm=&sf=&so=d&rg=50&c=Voting+Data&of=hb&fti=0&fct__2=Security+Council&fct__8=SOMALIA%20SITUATION&fti=0",
}

# ----------------- Scraping Functions -----------------
def get_voting_records(topic_name, topic_url, max_pages=10):
    """
    Scrape voting records for a given topic.

    :param topic_name: Name of the topic.
    :param topic_url: URL of the first page of the topic.
    :param max_pages: Maximum number of pages to scrape.
    :return: List of voting record URLs.
    """
    voting_records = []
    print(f"\n🔎 Scraping topic: {topic_name}")

    for page_number in range(max_pages):
        page_url = f"{topic_url}&jrec={page_number * 50 + 1}"  # Adjust for pagination
        print(f"📥 Fetching page: {page_url}")

        response = session.get(page_url, headers=HEADERS, timeout=15)
        soup = BeautifulSoup(response.text, "html.parser")

        record_links = soup.find_all("div", class_="moreinfo")
        if not record_links:
            print(f"⚠️ No voting records found for {topic_name}.")
            break  # Stop if no data is found

        for link in record_links:
            detailed_link_tag = link.find("a", class_="moreinfo", href=True)
            if detailed_link_tag and "record" in detailed_link_tag["href"]:
                record_url = BASE_URL + detailed_link_tag["href"]
                voting_records.append(record_url)

        print(f"✅ Retrieved {len(record_links)} records from {topic_name}.")
        time.sleep(random.uniform(1, 3))  # Avoid excessive requests

    return voting_records

def get_meeting_record(voting_url):
    """
    Parse the voting record page to find the meeting record details page URL.
    
    :param voting_url: The URL of a specific voting record page.
    :return: URL of the meeting record page or None if not found.
    """
    response = session.get(voting_url, headers=HEADERS, timeout=15)
    soup = BeautifulSoup(response.text, "html.parser")

    # Look for metadata-row containing "Meeting record"
    for row in soup.find_all("div", class_="metadata-row"):
        title_tag = row.find("span", class_="title")
        if title_tag and "Meeting record" in title_tag.text:
            value_tag = row.find("span", class_="value")
            if value_tag:
                meeting_record_link = value_tag.find("a", href=True)
                if meeting_record_link:
                    return meeting_record_link["href"]

    # Save HTML for debugging if no meeting record is found
    clean_filename = re.sub(r"[^a-zA-Z0-9_-]", "_", voting_url.split("/")[-1]) + ".html"
    debug_filename = f"debug_meeting_record_{clean_filename}"

    with open(debug_filename, "w", encoding="utf-8") as f:
        f.write(soup.prettify())
    print(f"⚠️ Meeting record page not found. Debugging HTML saved: {debug_filename}")

    return None

def get_pdf_link(meeting_record_url):
    """
    Parse the meeting record page to find the English PDF download link.
    
    :param meeting_record_url: URL of the meeting record page.
    :return: Direct link to the English PDF file or None if not found.
    """
    response = session.get(meeting_record_url, headers=HEADERS, timeout=15)
    soup = BeautifulSoup(response.text, "html.parser")

    for link in soup.find_all("a", href=True):
        if "-EN.pdf" in link["href"]:
            return BASE_URL + link["href"]

    return None

def download_pdf(pdf_url, filename):
    """
    Download the PDF file and save it locally.
    
    :param pdf_url: The direct URL to the PDF file.
    :param filename: The local filename to save the file as.
    :return: True if successful, False otherwise.
    """
    try:
        filepath = os.path.join(DOWNLOAD_DIR, filename)
        if os.path.exists(filepath):
            print(f"🟢 File already exists: {filename}")
            return True

        with session.get(pdf_url, headers=HEADERS, stream=True, timeout=30) as r:
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
        print(f"❌ Download failed: {str(e)}")
        return False

def main():
    """
    Main process: Iterate over different topics, scrape voting records, and download PDFs.
    """
    print("📌 Starting UN Security Council voting record scraper")

    for topic_name, topic_url in TOPICS.items():
        voting_records = get_voting_records(topic_name, topic_url, max_pages=10)

        for voting_url in tqdm(voting_records, desc=f"📊 Processing {topic_name}"):
            print(f"\n🔍 Processing voting record: {voting_url}")

            # Fetch meeting record URL
            meeting_record_url = get_meeting_record(voting_url)
            if not meeting_record_url:
                print(f"⚠️ Meeting record page not found. Skipping {voting_url}")
                continue

            print(f"📑 Meeting record page: {meeting_record_url}")

            # Fetch PDF link
            pdf_url = get_pdf_link(meeting_record_url)
            if not pdf_url:
                print(f"⚠️ English PDF not found. Skipping {meeting_record_url}")
                continue

            print(f"📄 PDF Download link: {pdf_url}")

            # Download PDF
            resolution_number = voting_url.split("/")[-1].split("?")[0]
            filename = f"{topic_name.replace(' ', '_')}_{resolution_number}.pdf"
            if download_pdf(pdf_url, filename):
                print(f"✅ Successfully downloaded: {filename}")
            else:
                print("❌ Download failed")

            time.sleep(random.uniform(2, 5))

if __name__ == "__main__":
    main()
