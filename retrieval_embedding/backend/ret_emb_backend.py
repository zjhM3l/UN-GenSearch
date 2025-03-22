import os
import pandas as pd
import numpy as np
import faiss
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from queue import Queue
import re

# ---------------------------
# 0. Logging for frontend
# ---------------------------
log_queue = Queue()

def strip_emoji(text):
    return re.sub(r"[^\x00-\x7F]+", "", text)

def log(msg):
    print(strip_emoji(msg))  # 控制台打印去除 emoji，防止 GBK 报错
    log_queue.put(msg)       # 原始消息（含 emoji）发送给前端

def get_log_updates():
    logs = []
    while not log_queue.empty():
        logs.append(log_queue.get())
    return logs

# ---------------------------
# 1. NLTK Resources
# ---------------------------
nltk.download('stopwords')
nltk.download('punkt_tab')
nltk.download('wordnet')

# ---------------------------
# 2. Cache Paths
# ---------------------------
CACHE_DIR = "./cache"
os.makedirs(CACHE_DIR, exist_ok=True)

DF_PATH = os.path.join(CACHE_DIR, "df.pkl")
TFIDF_PATH = os.path.join(CACHE_DIR, "tfidf_vectorizer.pkl")
TFIDF_MATRIX_PATH = os.path.join(CACHE_DIR, "tfidf_matrix.pkl")
FAISS_INDEX_PATH = os.path.join(CACHE_DIR, "faiss.index")
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# ---------------------------
# 3. Load or Build Indexes
# ---------------------------
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    if pd.isna(text): return ""
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word.isalnum()]
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return " ".join(tokens)

if os.path.exists(DF_PATH) and os.path.exists(TFIDF_PATH) and os.path.exists(TFIDF_MATRIX_PATH) and os.path.exists(FAISS_INDEX_PATH):
    log("✅ Cached indexes found. Loading from disk...")
    df = joblib.load(DF_PATH)
    tfidf_vectorizer = joblib.load(TFIDF_PATH)
    tfidf_matrix = joblib.load(TFIDF_MATRIX_PATH)
    faiss_index = faiss.read_index(FAISS_INDEX_PATH)
    log("✅ Indexes successfully loaded.")
else:
    log("🤖 First-time setup: Building index from scratch. This may take a few minutes...")

    log("🧠 Preprocessing and merging corpus...")
    df_raw = pd.read_csv("./datasets.csv", encoding="utf-8", encoding_errors="ignore")

    df_cover = df_raw[df_raw["type"] == 0].copy()
    df_cover["cover_text"] = df_cover.apply(
        lambda row: f"{row['s/pv']} {row['year']} {row['meeting_number']} {row['day_date_time']} {row['president']} {row['members']} {row['agenda']}", axis=1)
    df_cover = df_cover[["id", "title", "cover_text", "agenda"]]

    df_text = df_raw[df_raw["type"] > 0][["title", "text"]].copy()
    df_text = df_text.groupby("title")["text"].apply(lambda x: " ".join(x.dropna())).reset_index()

    df_merged = df_cover.merge(df_text, on="title", how="left").fillna("")
    df_merged["full_text"] = df_merged["cover_text"] + " " + df_merged["text"]
    df_merged["processed_text"] = df_merged["full_text"].apply(preprocess_text)
    df = df_merged

    log("📄 Preprocessing complete. Saving processed data...")
    joblib.dump(df, DF_PATH)

    log("🔧 Building TF-IDF index...")
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(df["processed_text"])
    tfidf_matrix = normalize(tfidf_matrix, norm='l2', axis=1)
    joblib.dump(tfidf_vectorizer, TFIDF_PATH)
    joblib.dump(tfidf_matrix, TFIDF_MATRIX_PATH)

    log("📦 Encoding documents with sentence embedding model...")
    embeddings = embedding_model.encode(df["processed_text"], convert_to_numpy=True, show_progress_bar=True)
    embeddings = normalize(embeddings, norm='l2', axis=1)

    log("🚀 Building FAISS index for semantic search...")
    dim = embeddings.shape[1]
    faiss_index = faiss.IndexFlatIP(dim)
    faiss_index.add(embeddings)
    faiss.write_index(faiss_index, FAISS_INDEX_PATH)

    log("✅ Indexing complete and cached! Ready to retrieve.")

# ---------------------------
# 4. Document Retrieval API
# ---------------------------
def retrieve_documents(query, top_k=5, alpha=0.6, beta=0.4):
    query_text = preprocess_text(query)
    query_tfidf = tfidf_vectorizer.transform([query_text])
    bm25_scores = np.dot(tfidf_matrix, query_tfidf.T).toarray().flatten()

    query_embedding = embedding_model.encode([query_text], convert_to_numpy=True)
    faiss_scores, faiss_indices = faiss_index.search(query_embedding, top_k)
    faiss_scores = faiss_scores.flatten()
    faiss_indices = faiss_indices.flatten()

    bm25_top_k_scores = bm25_scores[faiss_indices]
    bm25_top_k_scores = (bm25_top_k_scores - bm25_top_k_scores.min()) / (bm25_top_k_scores.max() - bm25_top_k_scores.min() + 1e-9)
    faiss_scores = (faiss_scores - faiss_scores.min()) / (faiss_scores.max() - faiss_scores.min() + 1e-9)

    final_scores = alpha * bm25_top_k_scores + beta * faiss_scores
    sorted_indices = np.argsort(final_scores)[::-1]
    top_indices = faiss_indices[sorted_indices]
    results = df.iloc[top_indices][["id", "title", "agenda", "text"]].to_dict(orient="records")
    return [entry["title"] for entry in results]
