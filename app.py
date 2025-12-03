import re
import json
from flask import Flask, render_template, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2
import docx
import io

app = Flask(__name__)

def clean_text(text):
    text = re.sub(r"[^a-zA-Z0-9 ]", " ", text)
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text

def calculate_relevancy(jd, resume):
    jd_clean = clean_text(jd)
    resume_clean = clean_text(resume)
    vectorizer = TfidfVectorizer(stop_words="english")
    vectors = vectorizer.fit_transform([jd_clean, resume_clean])
    score = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
    relevancy_percent = round(score * 100, 2)
    return relevancy_percent

def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""
    return text

def extract_text_from_docx(file):
    doc = docx.Document(file)
    text = "\n".join([para.text for para in doc.paragraphs])
    return text

def extract_text_from_file(file):
    filename = file.filename.lower()
    if filename.endswith('.pdf'):
        return extract_text_from_pdf(file)
    elif filename.endswith('.docx'):
        return extract_text_from_docx(file)
    elif filename.endswith('.txt'):
        return file.read().decode('utf-8')
    else:
        raise ValueError("Unsupported file format. Please upload PDF, DOCX, or TXT.")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        job_description = request.form.get('job_description', '')
        
        if not job_description.strip():
            return jsonify({'error': 'Please provide a job description'}), 400
        
        if 'resume' not in request.files:
            return jsonify({'error': 'Please upload a resume file'}), 400
        
        resume_file = request.files['resume']
        if resume_file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        resume_text = extract_text_from_file(resume_file)
        
        if not resume_text.strip():
            return jsonify({'error': 'Could not extract text from resume'}), 400
        
        relevancy = calculate_relevancy(job_description, resume_text)
        
        if relevancy > 70:
            category = "Strong Match"
            color = "#22c55e"
        elif relevancy > 40:
            category = "Moderate Match"
            color = "#f59e0b"
        else:
            category = "Weak Match"
            color = "#ef4444"
        
        return jsonify({
            'relevancy': relevancy,
            'category': category,
            'color': color
        })
    
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
