import os
import pandas as pd
import numpy as np
import faiss
import joblib
import nltk
import threading
from queue import Queue
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
import re
from time import time

# Logging queue for frontend
log_queue = Queue()
is_index_ready = False
index_lock = threading.Lock()
progress_state = {"step": 0}  # progress estimate，step=0-5

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def strip_emoji(text):
    return re.sub(r"[^\x00-\x7F]+", "", text)

def log(msg):
    print(strip_emoji(msg))
    log_queue.put(msg)

def get_log_updates():
    logs = []
    while not log_queue.empty():
        logs.append(log_queue.get())
    return logs

# NLTK setup
nltk.download('stopwords')
nltk.download('punkt_tab')
# nltk.download('punkt')
nltk.download('wordnet')

# Cache setup
CACHE_DIR = "./cache"
# CACHE_DIR = "./cache_old"
os.makedirs(CACHE_DIR, exist_ok=True)

DF_PATH = os.path.join(CACHE_DIR, "df.pkl")
TFIDF_PATH = os.path.join(CACHE_DIR, "tfidf_vectorizer.pkl")
TFIDF_MATRIX_PATH = os.path.join(CACHE_DIR, "tfidf_matrix.pkl")
FAISS_INDEX_PATH = os.path.join(CACHE_DIR, "faiss.index")
BATCH_SIZE = 2
# EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_MODEL_NAME = "Snowflake/snowflake-arctic-embed-l-v2.0"
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME).cuda()
# embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME, trust_remote_code=True).cuda()

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Globals
df = None
tfidf_vectorizer = None
tfidf_matrix = None
faiss_index = None

def preprocess_text(text):
    if pd.isna(text): return ""
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word.isalnum()]
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return " ".join(tokens)


def build_index():
    global df, tfidf_vectorizer, tfidf_matrix, faiss_index, is_index_ready

    with index_lock:
        try:
            if os.path.exists(DF_PATH) and os.path.exists(TFIDF_PATH) and os.path.exists(
                    TFIDF_MATRIX_PATH) and os.path.exists(FAISS_INDEX_PATH):
                log("✅ Cached indexes found. Loading from disk...")
                df = joblib.load(DF_PATH)
                tfidf_vectorizer = joblib.load(TFIDF_PATH)
                tfidf_matrix = joblib.load(TFIDF_MATRIX_PATH)
                faiss_index = faiss.read_index(FAISS_INDEX_PATH)
                log("✅ Indexes successfully loaded.")
            else:
                log("🤖 First-time setup: Building index from scratch...")

                # -------------------------------
                # Step 1: 读取原始数据并区分封面和正文记录
                # -------------------------------
                log("🧠 Step 1/5: Reading and preprocessing corpus...")
                progress_state["step"] = 1
                df_raw = pd.read_csv("./datasets.csv", encoding="utf-8", encoding_errors="ignore")

                # 提取封面记录：type == 0
                df_cover = df_raw[df_raw["type"] == 0].copy()
                # 为方便后续查找，将封面信息拼接成一个字符串
                df_cover["cover_text"] = df_cover.apply(
                    lambda row: f"{row['year']} {row['day_date_time']} {row['president']} {row['members']} {row['agenda']}",
                    axis=1)
                # 保留需要用到的字段，后续根据 title 对应封面记录查找
                df_cover = df_cover[["title", "cover_text"]]

                # 提取正文记录：type > 0
                df_body = df_raw[df_raw["type"] > 0].copy()

                # -------------------------------
                # Step 2: 对每个正文记录（type>0）查找其对应的封面信息，并组合文本
                # -------------------------------
                log("📄 Step 2/5: Merging cover info with body text...")
                progress_state["step"] = 2

                def combine_cover_and_text(row):
                    title = row["title"]
                    # 在封面数据中查找与当前 title 对应的记录
                    cover_row = df_cover[df_cover["title"] == title]
                    if not cover_row.empty:
                        cover_text = cover_row.iloc[0]["cover_text"]
                    else:
                        cover_text = ""
                    # 将封面信息与正文 text 拼接，这里返回的即合并后的文本
                    return f"{cover_text} {row['text']}".strip()

                # 为每一条正文记录生成 merged_text，并进行预处理
                df_body["merged_text"] = df_body.apply(combine_cover_and_text, axis=1)
                df_body["processed_text"] = df_body["merged_text"].apply(preprocess_text)
                # 注意：这里 df 只保留 type>0 的记录，不再进行按 title 聚合
                df = df_body.reset_index(drop=True)

                # 将处理后的 DataFrame 缓存到磁盘
                joblib.dump(df, DF_PATH)

                # -------------------------------
                # Step 3: 建立 TF-IDF 向量化器与矩阵
                # -------------------------------
                log("🔧 Step 3/5: Building TF-IDF index...")
                progress_state["step"] = 3
                tfidf_vectorizer = TfidfVectorizer()
                tfidf_matrix = tfidf_vectorizer.fit_transform(df["processed_text"])
                # 标准化 TF-IDF 特征
                tfidf_matrix = normalize(tfidf_matrix, norm='l2', axis=1)
                joblib.dump(tfidf_vectorizer, TFIDF_PATH)
                joblib.dump(tfidf_matrix, TFIDF_MATRIX_PATH)

                # -------------------------------
                # Step 4 & 5: 批量编码并增量构建 FAISS 索引
                # -------------------------------
                log("📦 Step 4/5: Encoding with embedding model and incrementally building FAISS index...")
                progress_state["step"] = 4
                texts = df["processed_text"].tolist()
                total = len(texts)
                print("total: ", total)
                start_time = time()

                # 先处理第一个批次，创建 FAISS 索引，得到维度信息
                first_batch = texts[0:BATCH_SIZE]
                first_embeddings = embedding_model.encode(first_batch, convert_to_numpy=True)
                first_embeddings = normalize(first_embeddings, norm='l2', axis=1)
                dim = first_embeddings.shape[1]
                faiss_index = faiss.IndexFlatIP(dim)
                faiss_index.add(first_embeddings)
                encoded_count = BATCH_SIZE

                # 每100次批处理后输出一次日志
                if (encoded_count // BATCH_SIZE) % 100 == 0:
                    elapsed = time() - start_time
                    log(f"🔹 Encoded {min(encoded_count, total)}/{total} texts in {elapsed:.1f}s")

                # 处理剩余的批次
                for i in range(BATCH_SIZE, total, BATCH_SIZE):
                    batch = texts[i:i + BATCH_SIZE]
                    batch_embeddings = embedding_model.encode(batch, convert_to_numpy=True)
                    batch_embeddings = normalize(batch_embeddings, norm='l2', axis=1)
                    faiss_index.add(batch_embeddings)
                    encoded_count += len(batch)
                    if (encoded_count // BATCH_SIZE) % 100 == 0:
                        elapsed = time() - start_time
                        log(f"🔹 Encoded {min(encoded_count, total)}/{total} texts in {elapsed:.1f}s")

                # -------------------------------
                # Step 5: 持久化 FAISS 索引
                # -------------------------------
                log("🚀 Step 5: Writing FAISS index to disk...")
                faiss.write_index(faiss_index, FAISS_INDEX_PATH)
                log("✅ Indexing complete and cached! Ready to retrieve.")

            is_index_ready = True
        except Exception as e:
            log(f"❌ Error during index building: {str(e)}")
            is_index_ready = False

def start_index_building_thread():
    threading.Thread(target=build_index, daemon=True).start()

def retrieve_documents(query, top_k=5, alpha=0.6, beta=0.4):
    global df, tfidf_vectorizer, tfidf_matrix, faiss_index
    if not is_index_ready:
        return ["⏳ Please wait, system is still preparing indexes..."]

    query_text = preprocess_text(query)
    query_tfidf = tfidf_vectorizer.transform([query_text])
    bm25_scores = np.dot(tfidf_matrix, query_tfidf.T).toarray().flatten()

    # query_embedding = embedding_model.encode([query_text], convert_to_numpy=True)
    query_embedding = embedding_model.encode([query_text], prompt_name="query", convert_to_numpy=True)
    faiss_scores, faiss_indices = faiss_index.search(query_embedding, top_k)
    faiss_scores = faiss_scores.flatten()
    faiss_indices = faiss_indices.flatten()

    bm25_top_k_scores = bm25_scores[faiss_indices]
    bm25_top_k_scores = (bm25_top_k_scores - bm25_top_k_scores.min()) / (
                bm25_top_k_scores.max() - bm25_top_k_scores.min() + 1e-9)
    faiss_scores = (faiss_scores - faiss_scores.min()) / (faiss_scores.max() - faiss_scores.min() + 1e-9)

    final_scores = alpha * bm25_top_k_scores + beta * faiss_scores
    sorted_indices = np.argsort(final_scores)[::-1]
    top_indices = faiss_indices[sorted_indices]
    results = df.iloc[top_indices][["id", "title", "agenda", "text"]].to_dict(orient="records")

    output_str = ""
    for entry in results:
        output_str += f"Document: {entry['title']}\n\n{entry['text']}\n\n"

    # print(output_str)
    with open("output.txt", "w", encoding="utf-8") as f:
        f.write(output_str)

    return [
        {"title": e["title"], "text": e["text"]}
        for e in results
    ]

