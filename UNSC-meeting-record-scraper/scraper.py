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

# 解决 Windows 终端编码问题
sys.stdout.reconfigure(encoding='utf-8')

# ----------------- 配置参数 -----------------
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

# ----------------- 重试配置 -----------------
retry_strategy = Retry(
    total=3,
    backoff_factor=1,
    status_forcelist=[429, 500, 502, 503, 504]
)
adapter = HTTPAdapter(max_retries=retry_strategy)
session = requests.Session()
session.mount("https://", adapter)

# ----------------- 获取所有表决记录 -----------------
def get_voting_records(max_pages=1):
    """
    爬取表决记录首页，获取每个会议详情页的 URL
    """
    voting_records = []
    for page_number in range(max_pages):
        page_url = VOTING_PAGE_URL_TEMPLATE.format(page_number * 50 + 1)
        print(f"📥 获取页面: {page_url}")
        response = session.get(page_url, headers=HEADERS, timeout=15)
        soup = BeautifulSoup(response.text, "html.parser")

        # 解析会议详情页链接
        record_links = soup.find_all("div", class_="moreinfo")
        if not record_links:
            print("⚠️ 没有找到表决记录，可能 HTML 结构发生变化！")
            break  # 没有数据，终止循环

        for link in record_links:
            detailed_link_tag = link.find("a", class_="moreinfo", href=True)
            if detailed_link_tag and "record" in detailed_link_tag["href"]:
                record_url = BASE_URL + detailed_link_tag["href"]
                voting_records.append(record_url)

        print(f"✅ 获取到 {len(record_links)} 条表决记录")
        time.sleep(random.uniform(1, 3))  # 避免请求过快

    return voting_records

# ----------------- 解析会议记录链接 -----------------
def get_meeting_record(voting_url):
    """
    解析表决记录页面，找到会议记录的详情页 URL
    """
    response = session.get(voting_url, headers=HEADERS, timeout=15)
    soup = BeautifulSoup(response.text, "html.parser")

    # 查找所有 metadata-row
    for row in soup.find_all("div", class_="metadata-row"):
        title_tag = row.find("span", class_="title")
        if title_tag and "Meeting record" in title_tag.text:
            value_tag = row.find("span", class_="value")
            if value_tag:
                meeting_record_link = value_tag.find("a", href=True)
                if meeting_record_link:
                    return meeting_record_link["href"]

    # 如果找不到，保存 HTML 供调试
    clean_filename = re.sub(r"[^a-zA-Z0-9_-]", "_", voting_url.split("/")[-1]) + ".html"
    debug_filename = f"debug_meeting_record_{clean_filename}"

    with open(debug_filename, "w", encoding="utf-8") as f:
        f.write(soup.prettify())
    print(f"⚠️ 未找到会议记录页面，已保存 HTML 供调试: {debug_filename}")

    return None

# ----------------- 解析会议记录 PDF -----------------
def get_pdf_link(meeting_record_url):
    """
    解析会议记录页面，找到英文版 PDF 的下载链接
    """
    response = session.get(meeting_record_url, headers=HEADERS, timeout=15)
    soup = BeautifulSoup(response.text, "html.parser")

    # 找到所有下载链接，筛选 English 版
    for link in soup.find_all("a", href=True):
        if "-EN.pdf" in link["href"]:  # 只抓取英文版 PDF
            return BASE_URL + link["href"]

    return None

# ----------------- 下载 PDF -----------------
def download_pdf(pdf_url, filename):
    """
    下载 PDF 文件，并存储到本地
    """
    try:
        filepath = os.path.join(DOWNLOAD_DIR, filename)
        if os.path.exists(filepath):
            print(f"🟢 文件已存在: {filename}")
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
        print(f"❌ 下载失败: {str(e)}")
        return False

# ----------------- 组合所有步骤 -----------------
def main():
    """
    主流程：获取表决记录 -> 进入会议记录 -> 下载 PDF
    """
    print("📌 开始爬取联合国安理会表决记录")

    # 只爬取前 1 页（50 条记录）
    voting_records = get_voting_records(max_pages=1)

    for voting_url in tqdm(voting_records, desc="📊 正在处理表决记录"):
        print(f"\n🔍 处理表决记录: {voting_url}")

        # 获取会议记录页面 URL
        meeting_record_url = get_meeting_record(voting_url)
        if not meeting_record_url:
            print(f"⚠️ 未找到会议记录页面，跳过 {voting_url}")
            continue

        print(f"📑 会议记录页面: {meeting_record_url}")

        # 获取 PDF 下载链接
        pdf_url = get_pdf_link(meeting_record_url)
        if not pdf_url:
            print(f"⚠️ 未找到英文版 PDF，跳过 {meeting_record_url}")
            continue

        print(f"📄 PDF 下载链接: {pdf_url}")

        # 下载 PDF
        resolution_number = voting_url.split("/")[-1].split("?")[0]  # 解析 URL 获取编号
        filename = f"{resolution_number}.pdf"
        if download_pdf(pdf_url, filename):
            print(f"✅ 下载成功: {filename}")
        else:
            print("❌ 下载失败")

        time.sleep(random.uniform(2, 5))  # 避免请求过快

if __name__ == "__main__":
    main()
