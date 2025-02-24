import PyPDF2
import re

def extract_qa_from_pdf(pdf_path):
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        text = "\n".join(filter(None, [page.extract_text() for page in reader.pages]))

    # Regex to extract Q&A
    qa_pairs = re.findall(r"Q:\s*(.*?)\s*A:\s*(.*?)(?=\nQ:|\Z)", text, re.DOTALL)

    qa_dict = {q.strip(): a.strip() for q, a in qa_pairs}
    return qa_dict

# ✅ Use a proper file path (No 'file:///')
pdf_path = "C:/Users/dell/Downloads/questionsanswers.pdf"
qa_data = extract_qa_from_pdf(pdf_path)

# ✅ Print extracted Q&A pairs
for question, answer in qa_data.items():
    print(f"Q: {question}\nA: {answer}\n")
