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
os.makedirs(CACHE_DIR, exist_ok=True)

DF_PATH = os.path.join(CACHE_DIR, "df.pkl")
TFIDF_PATH = os.path.join(CACHE_DIR, "tfidf_vectorizer.pkl")
TFIDF_MATRIX_PATH = os.path.join(CACHE_DIR, "tfidf_matrix.pkl")
FAISS_INDEX_PATH = os.path.join(CACHE_DIR, "faiss.index")
QUERY_PROMPT_NAME = "s2p_query"
BATCH_SIZE = 1
# EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_MODEL_NAME = "dunzhang/stella_en_1.5B_v5"
# embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME, trust_remote_code=True).cuda()

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
            if os.path.exists(DF_PATH) and os.path.exists(TFIDF_PATH) and os.path.exists(TFIDF_MATRIX_PATH) and os.path.exists(FAISS_INDEX_PATH):
                log("✅ Cached indexes found. Loading from disk...")
                df = joblib.load(DF_PATH)
                tfidf_vectorizer = joblib.load(TFIDF_PATH)
                tfidf_matrix = joblib.load(TFIDF_MATRIX_PATH)
                faiss_index = faiss.read_index(FAISS_INDEX_PATH)
                log("✅ Indexes successfully loaded.")
            else:
                log("🤖 First-time setup: Building index from scratch...")
                log("🧠 Step 1/5: Reading and preprocessing corpus...")
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

                log("📦 Step 4/5: Encoding with embedding model...")
                # progress_state["step"] = 4
                # # embeddings = embedding_model.encode(df["processed_text"], convert_to_numpy=True, show_progress_bar=True)
                # embeddings = embedding_model.encode(df["processed_text"].tolist(), prompt_name=QUERY_PROMPT_NAME,
                #                                     convert_to_numpy=True)
                #
                # embeddings = normalize(embeddings, norm='l2', axis=1)

                texts = df["processed_text"].tolist()
                total = len(texts)
                print("total: ", total)
                start_time = time()

                all_embeddings = []

                for i in range(0, total, BATCH_SIZE):
                    batch = texts[i:i + BATCH_SIZE]
                    batch_embeddings = embedding_model.encode(batch, prompt_name=QUERY_PROMPT_NAME,
                                                              convert_to_numpy=True)
                    all_embeddings.append(batch_embeddings)

                    elapsed = time() - start_time
                    log(f"🔹 Encoded {min(i + BATCH_SIZE, total)}/{total} texts in {elapsed:.1f}s")

                embeddings = np.vstack(all_embeddings)
                embeddings = normalize(embeddings, norm='l2', axis=1)

                log("🚀 Step 5/5: Building FAISS index...")
                progress_state["step"] = 5
                dim = embeddings.shape[1]
                faiss_index = faiss.IndexFlatIP(dim)
                faiss_index.add(embeddings)
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

    query_embedding = embedding_model.encode([query_text], prompt_name=QUERY_PROMPT_NAME, convert_to_numpy=True)
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

    print(output_str)
    with open("output.txt", "w", encoding="utf-8") as f:
        f.write(output_str)

    return [entry["title"] for entry in results]

