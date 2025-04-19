import os
import requests
from dotenv import load_dotenv
from flask import Flask, request, jsonify, send_from_directory
from backend import ret_emb_backend

# Load environment variables
load_dotenv()

# Deepseek API configuration
deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
deepseek_api_url = "https://api.deepseek.com/v1/chat/completions"

headers = {
    "Authorization": f"Bearer {deepseek_api_key}",
    "Content-Type": "application/json"
}


def call_deepseek_api(messages, model="deepseek-chat", temperature=0.2, max_tokens=1024):
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    try:
        resp = requests.post(deepseek_api_url, headers=headers, json=payload)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.RequestException as e:
        print(f"Deepseek API call failed: {e}")
        return None


app = Flask(__name__, static_folder="frontend", static_url_path="")
TOP_K = 5


@app.route("/", methods=["GET"])
def index():
    return send_from_directory("frontend", "retriever.html")


@app.route("/retrieve", methods=["POST"])
def retrieve():
    data = request.get_json()
    query = data.get("query", "")
    docs = ret_emb_backend.retrieve_documents(query, top_k=TOP_K)
    return jsonify({"documents": docs})


@app.route("/status", methods=["GET"])
def status():
    logs = ret_emb_backend.get_log_updates()
    return jsonify({
        "logs": logs,
        "progress": ret_emb_backend.progress_state["step"]
    })


@app.route("/answer", methods=["POST"])
def answer():
    data = request.get_json()
    query = data.get("query", "")
    docs = ret_emb_backend.retrieve_documents(query, top_k=TOP_K)

    # 提取文档标题列表
    doc_titles = [doc["title"] for doc in docs]

    # Assemble prompt in English
    context = [
        f"Document {i + 1} (Title: {doc['title']}):\n{doc['text']}"
        for i, doc in enumerate(docs)
    ]
    context_str = "\n\n".join(context)
    prompt = (
        f"Below are the documents relevant to your question: \"{query}\".\n\n"
        f"{context_str}\n\n"
        f"Based on the above documents and your own knowledge/databases, please provide a clear and concise answer to: {query}"
    )

    # Call Deepseek API
    messages = [
        {
            "role": "system",
            "content": (
                "You are an impartial expert assistant with deep knowledge in legal, international affairs, "
                "and ethical analysis of United Nations documents. Provide objective, comprehensive, "
                "and well-referenced answers based solely on the provided materials."
            )
        },
        {"role": "user", "content": prompt}
    ]
    response = call_deepseek_api(messages)
    if not response or "choices" not in response:
        return jsonify({"error": "Failed to get response from Deepseek API"}), 500

    answer_text = response["choices"][0]["message"]["content"].strip()
    # 在答案末尾添加文档来源
    formatted_answer = (
            f"{answer_text}\n\n"
            "📚 Reference Documents:\n" +
            "\n".join([f"- {title}" for title in doc_titles])
    )

    return jsonify({
        "answer": formatted_answer,
    })


if __name__ == "__main__":
    ret_emb_backend.start_index_building_thread()
    app.run(debug=True)
