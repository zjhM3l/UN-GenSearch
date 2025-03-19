import pandas as pd
import numpy as np
import faiss
import json
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

# 下载必要的NLTK资源
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# ---------------------------
# 1. 读取 CSV 并预处理数据
# ---------------------------

df = pd.read_csv("meetings.csv")  # 确保CSV文件已生成

# 停用词 & 词形归一化
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    tokens = word_tokenize(text.lower())  # 统一小写 & 分词
    tokens = [word for word in tokens if word.isalnum()]  # 去掉非字母字符
    tokens = [word for word in tokens if word not in stop_words]  # 去停用词
    tokens = [lemmatizer.lemmatize(word) for word in tokens]  # 词形还原
    return " ".join(tokens)

df["processed_text"] = df["text"].fillna("").apply(preprocess_text)

# ---------------------------
# 2. 构建索引（BM25 + FAISS）
# ---------------------------

# **(1) 构建 TF-IDF 索引**
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(df["processed_text"])
tfidf_matrix = normalize(tfidf_matrix, norm='l2', axis=1)  # 归一化

# **(2) 构建 FAISS 语义索引**
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
embeddings = embedding_model.encode(df["processed_text"], convert_to_numpy=True)
embeddings = normalize(embeddings, norm='l2', axis=1)  # 归一化

dim = embeddings.shape[1]
faiss_index = faiss.IndexFlatIP(dim)  # 使用内积相似度
faiss_index.add(embeddings)

# ---------------------------
# 3. 检索函数（Hybrid Search）
# ---------------------------

def retrieve_documents(query, top_k=5, alpha=0.6, beta=0.4):
    """
    执行混合搜索（Hybrid Search）：结合 BM25（关键词匹配）和 FAISS（语义搜索）。
    """
    # **(1) 预处理查询**
    query_text = preprocess_text(query)
    
    # **(2) 计算 BM25 相似度**
    query_tfidf = tfidf_vectorizer.transform([query_text])
    bm25_scores = np.dot(tfidf_matrix, query_tfidf.T).toarray().flatten()
    
    # **(3) 计算 FAISS 语义相似度**
    query_embedding = embedding_model.encode([query_text], convert_to_numpy=True)
    faiss_scores, faiss_indices = faiss_index.search(query_embedding, top_k)
    faiss_scores = faiss_scores.flatten()
    
    # **(4) 归一化 BM25 & FAISS 结果**
    bm25_scores = (bm25_scores - bm25_scores.min()) / (bm25_scores.max() - bm25_scores.min() + 1e-9)
    faiss_scores = (faiss_scores - faiss_scores.min()) / (faiss_scores.max() - faiss_scores.min() + 1e-9)
    
    # **(5) 综合排名**
    final_scores = alpha * bm25_scores + beta * faiss_scores
    top_indices = np.argsort(final_scores)[::-1][:top_k]
    
    # **(6) 返回查询结果**
    results = df.iloc[top_indices][["id", "title", "agenda", "text"]].to_dict(orient="records")
    return results

# ---------------------------
# 4. 查询示例
# ---------------------------

query = "Security Council nuclear disarmament"
top_k_results = retrieve_documents(query, top_k=5)

# **格式化 JSON 输出**
print(json.dumps(top_k_results, indent=4))
