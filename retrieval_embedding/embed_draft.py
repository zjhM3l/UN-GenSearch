import numpy as np
import faiss
import json
import pandas as pd
from sentence_transformers import SentenceTransformer

# ---------------------------
# 1. 选择 & 加载嵌入模型
# ---------------------------

# 选择合适的文本嵌入模型（可替换成更高性能模型）
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")  # 适合 CPU

# ---------------------------
# 2. 读取数据 & 预处理
# ---------------------------

df = pd.read_csv("meetings.csv")  # 确保 CSV 文件已生成
df["combined_text"] = df["title"].fillna("") + " " + df["text"].fillna("")  # 合并标题 + 内容

# ---------------------------
# 3. 计算 & 存储文本向量
# ---------------------------

embeddings = model.encode(df["combined_text"].tolist(), convert_to_numpy=True)
embeddings = np.array(embeddings, dtype=np.float32)  # FAISS 需要 float32 格式

# 创建 FAISS 索引
dim = embeddings.shape[1]  # 维度
faiss_index = faiss.IndexFlatIP(dim)  # 内积相似度（余弦相似度）
faiss_index.add(embeddings)

# ---------------------------
# 4. 相似度计算 & 搜索
# ---------------------------

def search_meetings(query, top_k=5):
    """
    输入查询文本，返回最相关的 Top-k 会议记录。
    """
    query_embedding = model.encode([query], convert_to_numpy=True).astype(np.float32)
    scores, indices = faiss_index.search(query_embedding, top_k)
    
    results = []
    for idx, score in zip(indices[0], scores[0]):
        results.append({
            "id": df.iloc[idx]["id"],
            "title": df.iloc[idx]["title"],
            "agenda": df.iloc[idx]["agenda"],
            "text": df.iloc[idx]["text"],
            "similarity": float(score)
        })
    
    return results

# ---------------------------
# 5. 查询示例
# ---------------------------

query = "Security Council nuclear disarmament"
top_k_results = search_meetings(query, top_k=5)

# **格式化 JSON 输出**
print(json.dumps(top_k_results, indent=4))
