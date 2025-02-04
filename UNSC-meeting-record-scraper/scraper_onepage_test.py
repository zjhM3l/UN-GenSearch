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

# 修复Windows终端编码问题
if sys.stdout.encoding != 'UTF-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

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

# ----------------- 调试配置 -----------------
DEBUG_MODE = True  # 开启调试模式
DEBUG_DIR = "./debug_html"
os.makedirs(DEBUG_DIR, exist_ok=True)

# ----------------- 重试配置 -----------------
retry_strategy = Retry(
    total=3,
    backoff_factor=1,
    status_forcelist=[429, 500, 502, 503, 504]
)
adapter = HTTPAdapter(max_retries=retry_strategy)
session = requests.Session()
session.mount("https://", adapter)

def save_debug_html(content, filename):
    """保存调试用HTML"""
    path = os.path.join(DEBUG_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"已保存调试文件: {path}")

def get_safe_soup(url):
    try:
        response = session.get(url, headers=HEADERS, timeout=15)
        response.raise_for_status()
        
        # 保存原始HTML用于调试
        if DEBUG_MODE:
            save_debug_html(response.text, f"raw_{url.split('/')[-1]}.html")
            
        return BeautifulSoup(response.text, "html.parser")
    except Exception as e:
        print(f"请求失败: {url} - {str(e)}")
        return None

def get_meeting_record(voting_url):
    soup = get_safe_soup(voting_url)
    if not soup:
        return None

    print("\n=== 新版定位逻辑 ===")
    
    # 调试1：打印页面标题
    title_tag = soup.find("title")
    print(f"页面标题: {title_tag.text if title_tag else '无标题'}")

    # 调试2：列出所有自定义组件
    custom_components = soup.find_all(lambda tag: tag.name.startswith("tindui-"))
    print(f"\n找到 {len(custom_components)} 个自定义组件:")
    for comp in custom_components:
        print(f"组件名: {comp.name}")
        print(f"属性: {comp.attrs}")
        print("-"*50)

    # 调试3：打印整个HTML结构的前1000字符
    print("\nHTML结构预览:")
    print(soup.prettify()[:1000])
    
    # 定位所有下载组件
    download_components = soup.find_all("tindui-app-file-download-link")
    print(f"\n找到 {len(download_components)} 个下载组件")

    # 如果找不到组件，尝试备用方案
    if not download_components:
        print("\n!!! 备用方案：尝试在原始HTML中搜索链接 !!!")
        all_links = soup.find_all("a", href=True)
        en_pdf_links = [link["href"] for link in all_links if "-EN.pdf" in link["href"]]
        print(f"找到 {len(en_pdf_links)} 个疑似英文PDF链接")
        if en_pdf_links:
            full_url = BASE_URL + en_pdf_links[0]
            print(f"备用链接: {full_url}")
            return full_url

    # 寻找包含英文的PDF链接
    for component in download_components:
        url = component.get("url", "")
        if "-EN.pdf" in url:
            full_url = BASE_URL + url
            print(f"找到英文PDF链接: {full_url}")
            return full_url

    print("!!! 未找到英文PDF组件 !!!")
    return None

# 需要新增的下载函数
def download_file(url, filename):
    try:
        filepath = os.path.join(DOWNLOAD_DIR, filename)
        if os.path.exists(filepath):
            print(f"文件已存在: {filename}")
            return True

        # 添加必要的请求头
        headers = {
            **HEADERS,
            "Referer": url  # 关键：需要携带来源头
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
        print(f"下载失败: {str(e)}")
        return False

def main():
    TEST_MODE = True
    records = ["https://digitallibrary.un.org/record/4031209"]
    
    print(f"待处理记录数: {len(records)}")

    for idx, voting_url in enumerate(records):
        print(f"\n=== 处理第 {idx+1}/{len(records)} 条记录 ===")
        print(f"表决页面: {voting_url}")

        # 直接获取PDF链接
        pdf_url = get_meeting_record(voting_url)
        print(f"PDF链接: {pdf_url}" if pdf_url else "未找到PDF链接")

        if not pdf_url:
            continue

        # 从URL提取文件名
        filename = pdf_url.split("/")[-1]
        
        print(f"开始下载: {filename}")
        if download_file(pdf_url, filename):
            print(f"下载成功: {filename}")
        else:
            print("下载失败")

if __name__ == "__main__":
    main()