from flask import Flask, request, jsonify, send_from_directory
from backend import ret_emb_backend
import os

app = Flask(__name__, static_folder="frontend", static_url_path="")

# Route: Serve retriever.html when accessing the root path "/"
@app.route("/")
def index():
    return send_from_directory("frontend", "retriever.html")

# Route: Serve static resources like .js or .css from /frontend/
@app.route("/frontend/<path:path>")
def serve_static(path):
    return send_from_directory("frontend", path)

# API: Handle document retrieval requests
@app.route("/retrieve", methods=["POST"])
def retrieve():
    data = request.get_json()
    query = data.get("query", "")
    titles = ret_emb_backend.retrieve_documents(query, top_k=5)
    return jsonify({"titles": titles})

# API: Frontend polling endpoint for logs and progress updates
@app.route("/status", methods=["GET"])
def status():
    logs = ret_emb_backend.get_log_updates()
    return jsonify({
        "logs": logs,
        "progress": ret_emb_backend.progress_state["step"]
    })

# Start background index building thread on app launch
if __name__ == "__main__":
    ret_emb_backend.start_index_building_thread()
    app.run(debug=True)
