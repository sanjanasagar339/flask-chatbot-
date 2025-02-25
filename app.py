from flask import Flask, request, jsonify
from flask_cors import CORS
import re
from fuzzywuzzy import process
from pdfminer.high_level import extract_text
import io

app = Flask(__name__)
CORS(app)

qa_data = {}

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

    normalized_qa_data = {q.lower(): a for q, a in qa_data.items()}
    if not normalized_qa_data:
        return jsonify({"answer": "No Q&A data available."})

    match = process.extractOne(user_question, normalized_qa_data.keys())
    if match is None or match[1] < 100:  # Adjust threshold as needed
        questions_list = list(qa_data.keys())
        return jsonify({
            "answer": "Irrelevant. Please try a question related to the topic.",
            "questions": questions_list
        })

    best_match, score = match
    return jsonify({"answer": normalized_qa_data[best_match]})
    
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
