import os
import re
import threading
from queue import Queue

import faiss
import joblib
import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

# ----------------------------
# Logging for frontend
# ----------------------------
log_queue = Queue()
is_index_ready = False
index_lock = threading.Lock()
progress_state = {"step": 0}

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

# ----------------------------
# Preprocessing Setup
# ----------------------------
nltk.download('stopwords')
nltk.download('punkt_tab')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    if pd.isna(text): return ""
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word.isalnum()]
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return " ".join(tokens)

# ----------------------------
# Chunking Utility
# ----------------------------
def chunk_text(text, chunk_size=128, stride=64):
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), stride) if i < len(words)]

# ----------------------------
# Cache Paths
# ----------------------------
CACHE_DIR = "./cache"
os.makedirs(CACHE_DIR, exist_ok=True)

DF_PATH = os.path.join(CACHE_DIR, "df.pkl")
TFIDF_PATH = os.path.join(CACHE_DIR, "tfidf_vectorizer.pkl")
TFIDF_MATRIX_PATH = os.path.join(CACHE_DIR, "tfidf_matrix.pkl")
FAISS_INDEX_PATH = os.path.join(CACHE_DIR, "faiss.index")
CHUNK_MAP_PATH = os.path.join(CACHE_DIR, "chunk_to_doc.npy")

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

# ----------------------------
# Global Objects
# ----------------------------
df = None
tfidf_vectorizer = None
tfidf_matrix = None
faiss_index = None
chunk_to_doc = None  # numpy array: chunk_idx -> doc_idx

# ----------------------------
# Index Building
# ----------------------------
def build_index():
    global df, tfidf_vectorizer, tfidf_matrix, faiss_index, is_index_ready, chunk_to_doc

    with index_lock:
        try:
            if all(os.path.exists(path) for path in [DF_PATH, TFIDF_PATH, TFIDF_MATRIX_PATH, FAISS_INDEX_PATH, CHUNK_MAP_PATH]):
                log("✅ Cached indexes found. Loading from disk...")
                df = joblib.load(DF_PATH)
                tfidf_vectorizer = joblib.load(TFIDF_PATH)
                tfidf_matrix = joblib.load(TFIDF_MATRIX_PATH)
                faiss_index = faiss.read_index(FAISS_INDEX_PATH)
                chunk_to_doc = np.load(CHUNK_MAP_PATH)
                log("✅ Indexes successfully loaded.")
            else:
                log("🤖 First-time setup: Building index from scratch...")
                progress_state["step"] = 1
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

                log("📄 Step 2/5: Preprocessing complete. Saving data...")
                progress_state["step"] = 2
                joblib.dump(df, DF_PATH)

                log("🔧 Step 3/5: Building TF-IDF index...")
                progress_state["step"] = 3
                tfidf_vectorizer = TfidfVectorizer()
                tfidf_matrix = tfidf_vectorizer.fit_transform(df["processed_text"])
                tfidf_matrix = normalize(tfidf_matrix, norm='l2', axis=1)
                joblib.dump(tfidf_vectorizer, TFIDF_PATH)
                joblib.dump(tfidf_matrix, TFIDF_MATRIX_PATH)

                log("📦 Step 4/5: Chunking and Encoding...")
                progress_state["step"] = 4
                all_chunks = []
                chunk_to_doc = []

                for idx, text in enumerate(df["processed_text"]):
                    chunks = chunk_text(text, chunk_size=128, stride=64)
                    all_chunks.extend(chunks)
                    chunk_to_doc.extend([idx] * len(chunks))

                chunk_embeddings = embedding_model.encode(all_chunks, convert_to_numpy=True, show_progress_bar=True)
                chunk_embeddings = normalize(chunk_embeddings, norm='l2', axis=1)
                chunk_to_doc = np.array(chunk_to_doc)

                log("🚀 Step 5/5: Building FAISS index over chunks...")
                progress_state["step"] = 5
                faiss_index = faiss.IndexFlatIP(chunk_embeddings.shape[1])
                faiss_index.add(chunk_embeddings)

                faiss.write_index(faiss_index, FAISS_INDEX_PATH)
                np.save(CHUNK_MAP_PATH, chunk_to_doc)

                log("✅ Indexing complete and cached! Ready to retrieve.")

            is_index_ready = True
        except Exception as e:
            log(f"❌ Error during index building: {str(e)}")
            is_index_ready = False

def start_index_building_thread():
    threading.Thread(target=build_index, daemon=True).start()

# ----------------------------
# Document Retrieval
# ----------------------------
def retrieve_documents(query, top_k=5, alpha=0.6, beta=0.4):
    global df, tfidf_vectorizer, tfidf_matrix, faiss_index, chunk_to_doc

    if not is_index_ready:
        return ["⏳ Please wait, system is still preparing indexes..."]

    query_text = preprocess_text(query)
    query_tfidf = tfidf_vectorizer.transform([query_text])
    bm25_scores = np.dot(tfidf_matrix, query_tfidf.T).toarray().flatten()

    query_embedding = embedding_model.encode([query_text], convert_to_numpy=True)
    faiss_scores, faiss_indices = faiss_index.search(query_embedding, 100)  # search top-100 chunks
    faiss_scores = faiss_scores.flatten()
    faiss_indices = faiss_indices.flatten()

    doc_scores = {}
    for chunk_idx, score in zip(faiss_indices, faiss_scores):
        doc_idx = chunk_to_doc[chunk_idx]
        if doc_idx not in doc_scores or score > doc_scores[doc_idx]:
            doc_scores[doc_idx] = score

    all_doc_ids = list(doc_scores.keys())
    dense_vals = np.array([doc_scores[i] for i in all_doc_ids])
    dense_vals = (dense_vals - dense_vals.min()) / (dense_vals.max() - dense_vals.min() + 1e-9)
    bm25_vals = bm25_scores[all_doc_ids]
    bm25_vals = (bm25_vals - bm25_vals.min()) / (bm25_vals.max() - bm25_vals.min() + 1e-9)

    final_scores = alpha * bm25_vals + beta * dense_vals
    sorted_indices = np.argsort(final_scores)[::-1]
    top_indices = [all_doc_ids[i] for i in sorted_indices[:top_k]]

    results = df.iloc[top_indices][["id", "title", "agenda"]].to_dict(orient="records")
    return [entry["title"] for entry in results]
