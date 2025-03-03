from flask import Flask, request, jsonify
from flask_cors import CORS
import re
import io
import os
from fuzzywuzzy import process
from pdfminer.high_level import extract_text
from sentence_transformers import SentenceTransformer, util

app = Flask(__name__)
CORS(app)

qa_data = {}
model = SentenceTransformer('all-MiniLM-L6-v2')  # Lightweight semantic similarity model

def extract_text_from_pdf(file_storage):
    pdf_bytes = file_storage.read()
    pdf_stream = io.BytesIO(pdf_bytes)
    extracted_text = extract_text(pdf_stream)
    return extracted_text.strip()

def extract_qa(text):
    qa_pairs = re.findall(r"Q:\s*(.*?)\s*\nA:\s*((?:.*?\n?)+?)(?=\nQ:|\Z)", text, re.DOTALL)
    return {q.strip(): a.strip().replace("\n", " ") for q, a in qa_pairs}

@app.route("/")
def home():
    return "Welcome to the chatbot API!"

@app.route("/upload", methods=["POST"])
def upload_pdf():
    global qa_data
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded."}), 400

    file = request.files["file"]
    extracted_text = extract_text_from_pdf(file)
    qa_data = extract_qa(extracted_text)
    return jsonify({"message": "Q&A extracted successfully!", "qa_count": len(qa_data)})

@app.route("/ask", methods=["POST"])
def ask_question():
    global qa_data
    data = request.json
    user_question = data.get("question", "").strip().lower()

    if not qa_data:
        return jsonify({"answer": "No Q&A data available."})

    topic_keywords = set()
    for q in qa_data.keys():
        topic_keywords.update(q.lower().split())

    question_words = set(user_question.split())
    relevant_words = question_words.intersection(topic_keywords)
    irrelevant_words = question_words - topic_keywords

    print(f"User Question: {user_question}")
    print(f"Relevant Words: {relevant_words}")
    print(f"Irrelevant Words: {irrelevant_words}")

    if len(irrelevant_words) >= len(relevant_words):
        return jsonify({
            "answer": "Irrelevant. Please try a question related to the topic.",
            "questions": list(qa_data.keys())
        })
    
    normalized_qa_data = {q.lower(): a for q, a in qa_data.items()}

    # Semantic matching with sentence embeddings
    question_embedding = model.encode(user_question, convert_to_tensor=True)
    qa_embeddings = {q: model.encode(q, convert_to_tensor=True) for q in normalized_qa_data.keys()}
    similarities = {q: util.pytorch_cos_sim(question_embedding, emb).item() for q, emb in qa_embeddings.items()}
    
    best_match = max(similarities, key=similarities.get)
    best_score = similarities[best_match]
    
    print(f"Best Match: {best_match} (Score: {best_score})")
    
    if best_score > 0.7:  # Adjust threshold for best results
        return jsonify({"answer": normalized_qa_data[best_match]})
    
    return jsonify({
        "answer": "Irrelevant. Please ask a more specific question about the provided topics.",
        "questions": list(qa_data.keys())
    })

if __name__ == "__main__":
    port = int(os.getenv("PORT", 10000))  # Ensure PORT is properly set
    app.run(host="0.0.0.0", port=port, debug=True)
