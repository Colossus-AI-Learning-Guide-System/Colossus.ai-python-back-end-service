from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import re

app = Flask(__name__)

# Check if MPS (Metal Performance Shaders) is available and set the device
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

# Load embedding model on the MPS device
embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)

# Load GPT model and move it to the MPS device
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-2.7B")
gpt_model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-2.7B")
gpt_model.to(device)

# Global store for uploaded documents
documents = []

@app.route('/upload', methods=['POST'])
def upload_document():
    global documents
    files = request.json.get("files", [])
    if not files:
        return jsonify({"error": "No files provided"}), 400

    for content in files:
        doc_embedding = embedding_model.encode(content, convert_to_tensor=True).to(device)
        documents.append({"content": content, "embedding": doc_embedding})

    return jsonify({"message": "Files uploaded successfully."}), 200

@app.route('/query', methods=['POST'])
def handle_query():
    query = request.json.get("query")
    if not query:
        return jsonify({"error": "Query is missing"}), 400
    if not documents:
        return jsonify({"error": "No documents to process"}), 400

    try:
        # Encode the query
        query_embedding = embedding_model.encode(query, convert_to_tensor=True).to(device)

        # Find the most relevant document based on cosine similarity
        top_doc = max(
            documents,
            key=lambda doc: util.pytorch_cos_sim(query_embedding, doc["embedding"]).item()
        )

        content = top_doc['content']
        lines = content.split("\n")
        graph_data = []
        current_title = None
        current_content = ""

        for line in lines:
            line = line.strip()
            if not line:
                continue  # Skip empty lines

            # Detect titles (flexible regex for numbers or bullet points)
            if re.match(r'^\d+\.', line) or re.match(r'^[IVXLCDM]+\.', line, re.IGNORECASE):
                if current_title:
                    graph_data.append({"title": current_title, "content": current_content.strip()})
                current_title = line.split(".", 1)[1].strip()
                current_content = ""
            else:
                current_content = f"{current_content} {line}".strip()

        # Append the last title-content pair
        if current_title:
            graph_data.append({"title": current_title, "content": current_content.strip()})

        # Fallback if no sections are detected
        if not graph_data:
            graph_data.append({"title": "Document Overview", "content": content.strip()})

        return jsonify(graph_data), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5001)