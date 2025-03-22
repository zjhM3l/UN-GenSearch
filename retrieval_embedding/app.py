from flask import Flask, request, jsonify
from backend import ret_emb_backend

app = Flask(__name__)

@app.route("/retrieve", methods=["POST"])
def retrieve():
    data = request.get_json()
    query = data.get("query", "")
    titles = ret_emb_backend.retrieve_documents(query, top_k=5)
    return jsonify({"titles": titles})

@app.route("/status", methods=["GET"])
def status():
    logs = ret_emb_backend.get_log_updates()
    return jsonify({"logs": logs})

if __name__ == "__main__":
    app.run(debug=True)
